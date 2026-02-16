# ===- fuse_ops.py -------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Construct op fusion pattern.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import *
from .. import DeviceType
from torch.fx.immutable_collections import immutable_list

classicfuse_register = {
    "transpose_matmul_fusion": TransposeMatmulFusedOp,
    "flash_attention_prefill_fusion": FlashAttentionForCpuPrefillOp,
    "gqa_attention_fusion": GQAAttentionFusedOp,
    "bias_add_fusion": MatmulWithAccOp,
}

# TODO: classify op type for op fusion
# OP_TYPE_FUSABLE = [OpType.BroadcastType, OpType.ElementwiseType, OpType.ReshapeType]
# OP_TYPE_UNFUSABLE = [OpType.Unfusable, OpType.ConcatType]
# OP_TYPE_FUSABLE_BY_SPECIFIC_PASS = []
# ANCHOR_OP_TYPE = []


def classic_fuse_check(graph: Graph):
    """
    Function to identifies and fuses PermuteOp operations with preceding
    MatmulOp operations in a computation graph to optimize performance.

    Args:
        graph (Graph): The computation graph to analyze and optimize.

    Returns:
        None
    """
    for op in graph.body:
        pattern = None
        if isinstance(op, MatmulOp):
            parentop = [graph.node_table[str(i)] for i in op._parents]
            for target in parentop:
                if isinstance(target, PermuteOp) and target.args[
                    1
                ] == immutable_list([1, 0]):
                    pattern = target, parentop, "transpose_matmul_fusion"
        if pattern:
            transpose_matmul_fusion(
                graph, op, pattern[0], pattern[1], pattern[2]
            )


def transpose_matmul_fusion(
    graph: Graph, node, target: Op, parents: List[Op], pattern: str
):
    """
    Function to fuse some typical operations into one operation.
    Such as transpose + matmul
    Args:
    - graph (Graph): The input graph to be simplified.
    - node (Op): The operation to be fused.
    - target (Op): The target operation to be fused.
    - parents (List[Op]): The parents of the node to be fused.
    - pattern (str): The pattern of the fusion.
    Returns:
    - None: Modifies the input graph in place.
    """
    fused_op = classicfuse_register.get(pattern)()
    # matmulop -> fusedmatmulopnode
    fused_op.name = "fused" + node.name
    graph.displace_node(node, fused_op)
    fused_op.args.pop(fused_op.args.index(target.name))
    fused_op._parents.pop(fused_op._parents.index(target.name))
    fused_op.args.extend(target.args)

    fused_op._parents.extend(target._parents)
    targets_parent = [graph.node_table[i] for i in target._parents]
    for i in targets_parent:
        i.add_children(fused_op.name)
    target._children.pop(target._children.index(fused_op.name))

    if graph.check_delete_node(target):
        graph.delete_node(target, targets_parent)


def bias_add_fuse_check(graph: Graph):
    """
    Function to detect and fuse matmul + add (bias) pattern.
    
    Pattern: MatmulOp -> AddOp (where add's other input is a 1D bias tensor)
    After fusion: MatmulWithAccOp (matmul with bias as accumulator)
    
    This fusion eliminates the separate add operation by using the bias
    as the initial accumulator value in the matmul operation.
    
    Note: This does NOT fuse residual connections where the other input is
    a large tensor (like the original input or a previous layer's output).
    """
    from ..operation import EmbeddingOp
    
    def get_shape_from_tensor_meta(tensor_meta):
        """Helper function to get shape from tensor_meta (can be dict or object)."""
        if tensor_meta is None:
            return None
        if isinstance(tensor_meta, dict):
            return tensor_meta.get('shape', None)
        elif hasattr(tensor_meta, 'shape'):
            return tensor_meta.shape
        return None
    
    # Collect all patterns first to avoid modifying graph during iteration
    patterns_to_fuse = []
    matmul_count = 0
    for op in list(graph.body):
        if isinstance(op, MatmulOp):
            matmul_count += 1
            # Debug: print info for first few matmuls
            if matmul_count <= 3:
                print(f"[DEBUG] MatmulOp {op.name}")
                for child_name in op._children:
                    child = graph.node_table.get(child_name, None)
                    if child and isinstance(child, (ViewOp, ReshapeOp)):
                        print(f"[DEBUG]   - child view '{child_name}' args: {child.args}")
                        for gc_name in child._children:
                            gc = graph.node_table.get(gc_name, None)
                            if gc and isinstance(gc, AddOp):
                                print(f"[DEBUG]     - add '{gc_name}' args: {gc.args}, parents: {gc._parents}")
                                for p_name in gc._parents:
                                    if p_name != child.name:
                                        p = graph.node_table.get(p_name, None)
                                        if p:
                                            print(f"[DEBUG]       - add parent '{p_name}' is {type(p).__name__}")
                                            print(f"[DEBUG]         - args: {p.args}")
                                            print(f"[DEBUG]         - tensor_meta: {p.tensor_meta}")
            
            # Check if matmul has a ViewOp/ReshapeOp as child
            for child_name in op._children:
                child = graph.node_table.get(child_name, None)
                if child is None:
                    continue
                # Pattern: MatmulOp -> ViewOp/ReshapeOp -> AddOp
                # But we need to check if the view is matmul's output, not bias's reshape
                if isinstance(child, (ViewOp, ReshapeOp)):
                    # Check if this view's input is the matmul output
                    if op.name in child._parents or op.name in child.args:
                        # This view is matmul's output reshape
                        # Check if this view/reshape has an AddOp as child
                        for grandchild_name in child._children:
                            grandchild = graph.node_table.get(grandchild_name, None)
                            if grandchild is None:
                                continue
                            if isinstance(grandchild, AddOp):
                                # Find the bias input (the one that's not the view/reshape output)
                                bias_parent = None
                                for parent_name in grandchild._parents:
                                    if parent_name != child.name:
                                        bias_parent = parent_name
                                        break
                                if bias_parent:
                                    # Check if bias_parent is a 1D tensor (bias) or a PlaceholderOp
                                    # We only want to fuse if it's a bias tensor, not a residual connection
                                    bias_node = graph.node_table.get(bias_parent)
                                    if bias_node:
                                        # Check if it's a PlaceholderOp with 1D shape (bias)
                                        # or a ReshapeOp/EmbeddingOp that reshapes a 1D tensor
                                        is_bias = False
                                        if isinstance(bias_node, PlaceholderOp):
                                            # Check tensor_meta for shape
                                            shape = get_shape_from_tensor_meta(bias_node.tensor_meta)
                                            if shape and len(shape) == 1:
                                                is_bias = True
                                        elif isinstance(bias_node, (ViewOp, ReshapeOp, EmbeddingOp)):
                                            # Check if this reshape's input is a 1D tensor
                                            for arg in bias_node.args:
                                                if isinstance(arg, str):
                                                    arg_node = graph.node_table.get(arg)
                                                    if arg_node and isinstance(arg_node, PlaceholderOp):
                                                        shape = get_shape_from_tensor_meta(arg_node.tensor_meta)
                                                        if shape and len(shape) == 1:
                                                            is_bias = True
                                                            break
                                        
                                        if is_bias:
                                            patterns_to_fuse.append((op, child, grandchild, bias_parent, "bias_add_fusion"))
                                    break
                        if patterns_to_fuse and patterns_to_fuse[-1][0] == op:
                            break
    
    print(f"[bias_add_fuse_check] Found {matmul_count} MatmulOp, {len(patterns_to_fuse)} patterns to fuse")
    
    # Apply all fusions after collecting patterns
    for matmul_op, view_op, add_op, bias_parent, pattern in patterns_to_fuse:
        bias_add_fusion(graph, matmul_op, view_op, add_op, bias_parent, pattern)


def bias_add_fusion(
    graph: Graph, 
    matmul_node: MatmulOp, 
    view_op: Op,
    add_op: AddOp, 
    bias_parent_name: str,
    pattern: str
):
    """
    Fuse matmul + add (bias) into MatmulWithAccOp.
    
    The pattern is: MatmulOp -> ViewOp/ReshapeOp -> AddOp
    After fusion: MatmulWithAccOp -> ViewOp/ReshapeOp
    
    The view/reshape operation is preserved for correct tensor shape handling.
    
    Args:
        graph: The computation graph
        matmul_node: The MatmulOp to be fused
        view_op: The ViewOp/ReshapeOp between matmul and add
        add_op: The AddOp that adds bias to matmul output
        bias_parent_name: Name of the bias tensor node
        pattern: The fusion pattern name
    """
    print(f"[bias_add_fusion] Fusing {matmul_node.name} + {add_op.name} with bias {bias_parent_name}")
    
    fuse_op = classicfuse_register.get(pattern)()
    fuse_op.name = "fused_" + matmul_node.name
    graph.displace_node(matmul_node, fuse_op)
    
    # Copy tensor metadata from original matmul
    if hasattr(matmul_node, 'tensor_meta') and matmul_node.tensor_meta:
        fuse_op.tensor_meta = matmul_node.tensor_meta.copy()
    
    # Add bias as the third argument to the fused op
    fuse_op._parents.append(bias_parent_name)
    fuse_op.args.append(bias_parent_name)
    
    print(f"[bias_add_fusion] fuse_op.name={fuse_op.name}, args={fuse_op.args}, _parents={fuse_op._parents}")
    
    # Update bias parent's children to point to fused op
    bias_parent = graph.node_table.get(bias_parent_name)
    if bias_parent:
        if add_op.name in bias_parent._children:
            idx = bias_parent._children.index(add_op.name)
            bias_parent._children[idx] = fuse_op.name
        else:
            bias_parent.add_children(fuse_op.name)
    
    # The fused op's output goes to view_op (preserved)
    fuse_op._children = [view_op.name]
    
    # Update view_op's parent to point to fused op
    if matmul_node.name in view_op._parents:
        idx = view_op._parents.index(matmul_node.name)
        view_op._parents[idx] = fuse_op.name
    if matmul_node.name in view_op.args:
        idx = view_op.args.index(matmul_node.name)
        view_op.args[idx] = fuse_op.name
    
    # Update add's children to use view_op's output (bypass add)
    add_children = [
        graph.node_table[child_name] 
        for child_name in add_op._children
    ]
    for child in add_children:
        if add_op.name in child._parents:
            parent_idx = child._parents.index(add_op.name)
            child._parents[parent_idx] = view_op.name
        
        if add_op.name in child.args:
            arg_idx = child.args.index(add_op.name)
            child.args[arg_idx] = view_op.name
    
    # Update view_op's children to point to add's children
    view_op._children = add_op._children.copy()
    
    # Clear add op's connections
    add_op._children.clear()
    add_op._parents.clear()
    
    # Delete the add op
    add_parents = []
    for parent_name in [view_op.name, bias_parent_name]:
        parent = graph.node_table.get(parent_name)
        if parent and add_op.name in parent._children:
            add_parents.append(parent)
    
    if graph.check_delete_node(add_op) and add_parents:
        graph.delete_node(add_op, add_parents)


def apply_classic_fusion(graph: Graph):
    """
    Function to fuse some typical operations into one operation and fuse
    all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    # Run the first round of op fusion
    classic_fuse_check(graph)
    bias_add_fuse_check(graph)
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}


def simply_fuse(graph: Graph):
    """
    Function to fuse all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}


def flash_attention_prefill(graph: Graph):
    """
    Replace ScaledDotProductFlashAttentionForCpuOp with FlashAttentionForCpuPrefillOp.
    """
    new_op_group = []
    device = DeviceType.CPU
    replace_attention_op(graph)

    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)

    graph.op_groups = {"subgraph0": new_op_group}
    graph.group_map_device = {"subgraph0": device}


def replace_attention_op(graph: Graph):
    """
    replace ScaledDotProductFlashAttentionForCpuOp with FlashAttentionForCpuPrefillOp.
    """
    for op in list(graph.body):
        if isinstance(op, ScaledDotProductFlashAttentionForCpuOp):
            new_op = classicfuse_register.get(
                "flash_attention_prefill_fusion"
            )()
            new_op.name = "FlashAttentionForCpuPrefillOp"
            graph.displace_node(op, new_op)


def gqa_attention_fusion(graph: Graph):
    """
    Function to fuse GQA Attention operations into one operation and fuse
    all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    gqa_attention_fusion_check(graph)
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}


def gqa_attention_fusion_check(graph: Graph):
    for op in graph.body:
        # === GQA Attention pattern ===
        if isinstance(op, ScaledDotProductFlashAttentionForCpuOp):

            # get KV and View nodes
            k_view_node = graph.node_table.get(op._parents[1], None)
            v_view_node = graph.node_table.get(op._parents[2], None)

            if not (
                isinstance(k_view_node, ViewOp)
                and isinstance(v_view_node, ViewOp)
            ):
                continue

            # trace Key branch: View <- Clone <- Expand <- slice1 <- slice2 <- unsqueeze
            k_clone = graph.node_table.get(k_view_node._parents[0], None)
            if not isinstance(k_clone, CloneOp):
                continue
            k_expand = graph.node_table.get(k_clone._parents[0], None)
            if not isinstance(k_expand, ExpandOp):
                continue
            k_slice1 = graph.node_table.get(k_expand._parents[0], None)
            if not isinstance(k_slice1, SliceOp):
                continue
            k_slice2 = graph.node_table.get(k_slice1._parents[0], None)
            if not isinstance(k_slice2, SliceOp):
                continue
            k_cache_unsqueeze = graph.node_table.get(k_slice2._parents[0], None)
            if not isinstance(k_cache_unsqueeze, UnsqueezeOp):
                continue

            # trace Value branch: View <- Clone <- Expand <- slice1 <- slice2 <- unsqueeze
            v_clone = graph.node_table.get(v_view_node._parents[0], None)
            if not isinstance(v_clone, CloneOp):
                continue
            v_expand = graph.node_table.get(v_clone._parents[0], None)
            if not isinstance(v_expand, ExpandOp):
                continue
            v_slice1 = graph.node_table.get(v_expand._parents[0], None)
            if not isinstance(v_slice1, SliceOp):
                continue
            v_slice2 = graph.node_table.get(v_slice1._parents[0], None)
            if not isinstance(v_slice2, SliceOp):
                continue
            v_cache_unsqueeze = graph.node_table.get(v_slice2._parents[0], None)
            if not isinstance(v_cache_unsqueeze, UnsqueezeOp):
                continue
            replace_gqa_attention_with_fused_op(
                graph,
                op,
                k_view_node,
                k_clone,
                k_expand,
                k_slice1,
                k_slice2,
                k_cache_unsqueeze,
                v_view_node,
                v_clone,
                v_expand,
                v_slice1,
                v_slice2,
                v_cache_unsqueeze,
                "gqa_attention_fusion",
            )


def replace_gqa_attention_with_fused_op(
    graph: Graph,
    sdpa_node: Op,
    k_view: Op,
    k_clone: Op,
    k_expand: Op,
    k_slice1: Op,
    k_slice2: Op,
    k_cache_unsqueeze: Op,
    v_view: Op,
    v_clone: Op,
    v_expand: Op,
    v_slice1: Op,
    v_slice2: Op,
    v_cache_unsqueeze: Op,
    pattern: str,
):
    """
    Fuse GQA subgraph
    into one GQAAttentionFusedOp.
    """
    fused_cls = classicfuse_register.get(pattern)
    fused_op = fused_cls()
    fused_op.name = "GQAAttentionFusedOp"

    # replace SDPA node with GQAAttentionFusedOp
    graph.displace_node(sdpa_node, fused_op)

    # clear old KV View input inherited by SDPA
    # assume sdpa_node.args[0] is Query, keep unchanged
    # args[1] and args[2] are k_view and v_view, need to pop
    fused_op.args.pop(fused_op.args.index(k_view.name))
    fused_op._parents.pop(fused_op._parents.index(k_view.name))
    fused_op.args.pop(fused_op.args.index(v_view.name))
    fused_op._parents.pop(fused_op._parents.index(v_view.name))

    for k_parent in k_cache_unsqueeze._parents:
        fused_op._parents.append(k_parent)
        fused_op.args.append(k_parent)
    for v_parent in v_cache_unsqueeze._parents:
        fused_op._parents.append(v_parent)
        fused_op.args.append(v_parent)

    k_view._children.clear()
    if graph.check_delete_node(k_view):
        graph.delete_node(k_view, [k_clone])
    if graph.check_delete_node(k_clone):
        graph.delete_node(k_clone, [k_expand])
    if graph.check_delete_node(k_expand):
        graph.delete_node(k_expand, [k_slice1])
    if graph.check_delete_node(k_slice1):
        graph.delete_node(k_slice1, [k_slice2])
    if graph.check_delete_node(k_slice2):
        graph.delete_node(k_slice2, [k_cache_unsqueeze])
    if graph.check_delete_node(k_cache_unsqueeze):
        k_orig_parents = [
            graph.node_table.get(p, None) for p in k_cache_unsqueeze._parents
        ]
        graph.delete_node(k_cache_unsqueeze, k_orig_parents)

    v_view._children.clear()
    if graph.check_delete_node(v_view):
        graph.delete_node(v_view, [v_clone])
    if graph.check_delete_node(v_clone):
        graph.delete_node(v_clone, [v_expand])
    if graph.check_delete_node(v_expand):
        graph.delete_node(v_expand, [v_slice1])
    if graph.check_delete_node(v_slice1):
        graph.delete_node(v_slice1, [v_slice2])
    if graph.check_delete_node(v_slice2):
        graph.delete_node(v_slice2, [v_cache_unsqueeze])
    if graph.check_delete_node(v_cache_unsqueeze):
        v_orig_parents = [
            graph.node_table.get(p, None) for p in v_cache_unsqueeze._parents
        ]
        graph.delete_node(v_cache_unsqueeze, v_orig_parents)
