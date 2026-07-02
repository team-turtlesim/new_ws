from datetime import datetime, timezone


def normalize_node_name(node_name, namespace='/'):
    namespace = str(namespace or '/')
    node_name = str(node_name or '').strip('/')
    if not node_name:
        return '/'
    if namespace == '/':
        return f'/{node_name}'
    return f"{namespace.rstrip('/')}/{node_name}"


def build_graph_snapshot(node):
    node_names = sorted({
        normalize_node_name(name)
        for name in node.get_node_names()
    })
    topic_info = list(node.get_topic_names_and_types())

    graph_nodes = [
        {
            'id': f'node:{node_name}',
            'label': node_name,
            'kind': 'node',
        }
        for node_name in node_names
    ]
    graph_edges = []
    known_ids = {item['id'] for item in graph_nodes}

    for topic_name, topic_types in topic_info:
        topic_id = f'topic:{topic_name}'
        graph_nodes.append({
            'id': topic_id,
            'label': topic_name,
            'kind': 'topic',
            'types': list(topic_types),
        })
        known_ids.add(topic_id)

        for publisher in node.get_publishers_info_by_topic(topic_name):
            node_name = normalize_node_name(
                publisher.node_name,
                publisher.node_namespace,
            )
            node_id = f'node:{node_name}'
            if node_id not in known_ids:
                graph_nodes.append({
                    'id': node_id,
                    'label': node_name,
                    'kind': 'node',
                })
                known_ids.add(node_id)
            graph_edges.append({
                'source': node_id,
                'target': topic_id,
                'topic': topic_name,
                'direction': 'publishes',
            })

        for subscriber in node.get_subscriptions_info_by_topic(topic_name):
            node_name = normalize_node_name(
                subscriber.node_name,
                subscriber.node_namespace,
            )
            node_id = f'node:{node_name}'
            if node_id not in known_ids:
                graph_nodes.append({
                    'id': node_id,
                    'label': node_name,
                    'kind': 'node',
                })
                known_ids.add(node_id)
            graph_edges.append({
                'source': topic_id,
                'target': node_id,
                'topic': topic_name,
                'direction': 'subscribes',
            })

    label_by_id = {item['id']: item['label'] for item in graph_nodes}
    graph_edges.sort(
        key=lambda edge: label_by_id.get(edge['source'], '').lstrip('/').lower()
    )

    return {
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'nodes': graph_nodes,
        'edges': graph_edges,
    }
