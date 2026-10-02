"""Preserve reference order for conflicting primary records in the frozen sample.

This guard covers primary user/order/product/reservation records. It is not a
general transaction dependency analyzer for arbitrary workflows or payment
accounts. Native locking additionally prevents lost updates to the shared DB.
"""
import copy


READ_ONLY = {
    'find_user_by_contact', 'find_user_by_name', 'get_user_details',
    'get_order_details', 'get_product_details',
}
MUTATIONS = {
    'modify_user_address', 'cancel_order', 'return_delivered_order_items',
    'exchange_delivered_order_items', 'modify_pending_order_items',
    'modify_pending_order_address', 'modify_pending_order_payment',
    'update_reservation_flights', 'update_reservation_baggages',
}


def primary_record(node):
    tool = node['entrypoint']
    if tool not in READ_ONLY | MUTATIONS:
        raise ValueError('No state conflict contract for ' + tool)
    params = {p['name']: p['value'] for p in node.get('parameters', [])}
    for key in ('order_id', 'reservation_id', 'user_id', 'product_id'):
        if key in params:
            kind = 'order' if key in ('order_id', 'reservation_id') else key
            return kind, str(params[key])
    return None


def derive_plan(original, topological_waves):
    """Return an execution copy and newly added, auditable control dependencies."""
    workflow = copy.deepcopy(original)
    nodes = {n['node_id']: n for n in workflow['nodes']}
    business = [n for n in workflow['nodes'] if n['kind'] in ('app', 'dnn')]
    added = []

    def reachable(source, target):
        todo, seen = [source], set()
        while todo:
            current = todo.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            todo.extend(e['to_node'] for e in workflow['edges']
                        if e['from_node'] == current)
        return False

    for i, first in enumerate(business):
        record = primary_record(first)
        for second in business[i + 1:]:
            if record is None or record != primary_record(second):
                continue
            if all(n['entrypoint'] in READ_ONLY for n in (first, second)):
                continue
            source, target = first['node_id'], second['node_id']
            if reachable(source, target):
                continue
            if reachable(target, source):
                raise ValueError('DAG reverses conflicting reference actions')
            checker = 'check_after_' + source
            if checker in nodes and reachable(source, checker):
                source = checker
            if reachable(target, source):
                raise ValueError('State conflict constraint would create a cycle')
            edge = {
                'from_node': source,
                'to_node': target,
                'reason': 'Preserve original action order for shared mutable record',
                'state_resource': list(record),
            }
            workflow['edges'].append(edge)
            added.append(copy.deepcopy(edge))
    if added:
        workflow['execution_hints']['parallel_waves'] = topological_waves(workflow)
        workflow.setdefault('metadata', {})['phone_state_conflict_dependencies'] = added
    return workflow, added
