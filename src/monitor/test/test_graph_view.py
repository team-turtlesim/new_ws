import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from monitor.flask_app_factory import create_app
from monitor.graph_utils import build_graph_snapshot


class FakeGraphNode:
    def get_node_names(self):
        return ['camera_node', 'monitor_node']

    def get_topic_names_and_types(self):
        return [
            ('/camera/image/compressed', ['sensor_msgs/msg/CompressedImage']),
        ]

    def get_publishers_info_by_topic(self, topic_name):
        publishers = {
            '/camera/image/compressed': [
                SimpleNamespace(node_name='camera_node', node_namespace='/'),
            ],
        }
        return publishers.get(topic_name, [])

    def get_subscriptions_info_by_topic(self, topic_name):
        subscribers = {
            '/camera/image/compressed': [
                SimpleNamespace(node_name='monitor_node', node_namespace='/'),
            ],
        }
        return subscribers.get(topic_name, [])


class FakeUnsortedGraphNode:
    def get_node_names(self):
        return ['camera_node', 'zed_node', 'battery_node']

    def get_topic_names_and_types(self):
        return [
            ('/camera/older', ['sensor_msgs/msg/CompressedImage']),
            ('/zed/status', ['std_msgs/msg/String']),
            ('/camera/newer', ['sensor_msgs/msg/CompressedImage']),
            ('/battery/state', ['std_msgs/msg/String']),
        ]

    def get_publishers_info_by_topic(self, topic_name):
        publishers = {
            '/camera/older': [
                SimpleNamespace(node_name='camera_node', node_namespace='/'),
            ],
            '/camera/newer': [
                SimpleNamespace(node_name='camera_node', node_namespace='/'),
            ],
            '/zed/status': [
                SimpleNamespace(node_name='zed_node', node_namespace='/'),
            ],
            '/battery/state': [
                SimpleNamespace(node_name='battery_node', node_namespace='/'),
            ],
        }
        return publishers.get(topic_name, [])

    def get_subscriptions_info_by_topic(self, _topic_name):
        return []


class FakeState:
    def snapshot(self):
        return {
            'battery': {},
            'image': {},
            'control': {},
            'recording': {},
            'storage': {},
        }

    def get_latest_frame(self):
        return None

    def get_debug_frame(self, _image_key):
        return None



def read_monitor_static_asset(*path_parts):
    package_root = Path(create_app.__code__.co_filename).resolve().parent
    candidates = [package_root / 'static' / Path(*path_parts)]
    for parent in Path(__file__).resolve().parents:
        candidates.append(
            parent / 'src' / 'monitor' / 'monitor' / 'static' / Path(*path_parts)
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding='utf-8')

    raise FileNotFoundError(Path('static') / Path(*path_parts))



def render_graph_html(payload):
    node_binary = shutil.which('node')
    if node_binary is None:
        raise RuntimeError('node is required to exercise the graph renderer')

    script = read_monitor_static_asset('js', 'app.js')
    runner = f"""
const vm = require('vm');
const script = {json.dumps(script)};
const payload = {json.dumps(payload)};
const elements = new Map();
function makeElement(id) {{
  return {{
    id,
    innerHTML: '',
    textContent: '',
    src: '',
    style: {{}},
    classList: {{
      remove() {{}},
      add() {{}},
      toggle() {{}},
    }},
  }};
}}
const context = {{
  console: {{ error() {{}}, log() {{}} }},
  window: {{ MONITOR_CONFIG: {{ graphEndpoint: '/api/graph' }}, setInterval() {{}} }},
  document: {{
    getElementById(id) {{
      if (!elements.has(id)) {{
        elements.set(id, makeElement(id));
      }}
      return elements.get(id);
    }},
    addEventListener() {{}},
  }},
  Image: function Image() {{}},
}};
vm.createContext(context);
vm.runInContext(`${{script}}\nwindow.__renderGraph = renderGraph;`, context);
context.window.__renderGraph(payload);
process.stdout.write(elements.get('ros-graph-canvas').innerHTML);
"""
    result = subprocess.run(
        [node_binary, '-e', runner],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout

def make_test_app(graph_snapshot_provider=None):
    resource_path = Path(__file__).resolve()
    return create_app(
        FakeState(),
        'Test Monitor',
        'battery_status',
        '/camera/image/compressed',
        '/control',
        '/',
        1000,
        100,
        resource_path,
        resource_path,
        resource_path,
        160,
        120,
        True,
        '/opencv/image/grayscale',
        '/opencv/image/blur',
        '/opencv/image/edge',
        graph_snapshot_provider=graph_snapshot_provider,
    )


def test_build_graph_snapshot_returns_nodes_topics_and_edges():
    graph = build_graph_snapshot(FakeGraphNode())

    assert graph['updated_at']
    assert {
        'id': 'node:/camera_node',
        'label': '/camera_node',
        'kind': 'node',
    } in graph['nodes']
    assert {
        'id': 'topic:/camera/image/compressed',
        'label': '/camera/image/compressed',
        'kind': 'topic',
        'types': ['sensor_msgs/msg/CompressedImage'],
    } in graph['nodes']
    assert {
        'source': 'node:/camera_node',
        'target': 'topic:/camera/image/compressed',
        'topic': '/camera/image/compressed',
        'direction': 'publishes',
    } in graph['edges']
    assert {
        'source': 'topic:/camera/image/compressed',
        'target': 'node:/monitor_node',
        'topic': '/camera/image/compressed',
        'direction': 'subscribes',
    } in graph['edges']



def test_build_graph_snapshot_orders_edges_by_left_name_then_original_order():
    graph = build_graph_snapshot(FakeUnsortedGraphNode())

    assert [edge['topic'] for edge in graph['edges']] == [
        '/battery/state',
        '/camera/older',
        '/camera/newer',
        '/zed/status',
    ]


def test_flask_graph_endpoint_returns_current_graph():
    payload = {
        'updated_at': '2026-06-19T00:00:00+00:00',
        'nodes': [{'id': 'node:/camera_node'}],
        'edges': [],
    }
    app = make_test_app(graph_snapshot_provider=lambda: payload)

    with app.test_request_context('/api/graph'):
        response = app.view_functions['api_graph']()

    assert response.get_json() == payload



def test_graph_renderer_groups_publish_edges_by_source_node():
    html = render_graph_html({
        'updated_at': '2026-06-19T00:00:00+00:00',
        'nodes': [
            {'id': 'node:/camera_node', 'label': '/camera_node', 'kind': 'node'},
            {'id': 'topic:/camera/older', 'label': '/camera/older', 'kind': 'topic'},
            {'id': 'topic:/camera/newer', 'label': '/camera/newer', 'kind': 'topic'},
            {'id': 'node:/monitor_node', 'label': '/monitor_node', 'kind': 'node'},
        ],
        'edges': [
            {
                'source': 'node:/camera_node',
                'target': 'topic:/camera/older',
                'topic': '/camera/older',
                'direction': 'publishes',
            },
            {
                'source': 'node:/camera_node',
                'target': 'topic:/camera/newer',
                'topic': '/camera/newer',
                'direction': 'publishes',
            },
            {
                'source': 'topic:/camera/older',
                'target': 'node:/monitor_node',
                'topic': '/camera/older',
                'direction': 'subscribes',
            },
        ],
    })

    assert html.count('ros-graph__publisher-group') == 1
    assert html.count('ros-graph__publisher-topic-row') == 2
    assert html.find('/camera/older') < html.find('/camera/newer')
    assert '/monitor_node' in html


def test_dashboard_includes_graph_view_assets():
    app = make_test_app(graph_snapshot_provider=lambda: {'nodes': [], 'edges': []})

    with app.test_request_context('/'):
        template = app.view_functions['index']()

    script = read_monitor_static_asset('js', 'app.js')
    stylesheet = read_monitor_static_asset('css', 'style.css')

    assert 'graphEndpoint' in template
    assert 'graph-card' in template
    assert 'ros-graph-canvas' in template
    assert 'graph-summary' in template
    assert 'fetchGraph' in script
    assert 'ros-graph__node-card' in script
    assert 'ros-graph__edge-label' in script
    assert '.graph-card' in stylesheet
    assert '.ros-graph__row' in stylesheet
    assert '.ros-graph__edge::before' in stylesheet
