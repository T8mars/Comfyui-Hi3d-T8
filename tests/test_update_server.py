from fastapi.testclient import TestClient


def test_update_scheduler_belongs_to_server_lifespan_not_gpu_operations(tmp_path, monkeypatch):
    from hi3d_workbench import server
    events = []
    monkeypatch.setattr(server, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(server.manager, 'recover', lambda: None)
    monkeypatch.setattr(server.manager, 'active', None)
    monkeypatch.setattr(server.manager, '_recovery_workers', [])
    monkeypatch.setattr(server, 'probe_recovery_workers', lambda: [])
    monkeypatch.setattr(server.models, 'status', lambda: {'ready': True})
    monkeypatch.setattr(server.pipeline_models, 'status', lambda: {'runtime': {'ready': False}})
    monkeypatch.setattr(server.updater, 'start_background', lambda busy: events.append(('start', busy)))
    monkeypatch.setattr(server.updater, 'stop_background', lambda: events.append(('stop', None)))
    with TestClient(server.app, base_url='http://127.0.0.1:8378') as client:
        assert client.get('/api/session').status_code == 200
        assert [event[0] for event in events] == ['start']
        assert not events[0][1]()
        with server.gpu_operation('test'):
            assert events[0][1]()
        assert [event[0] for event in events] == ['start']
        assert not events[0][1]()
    assert [event[0] for event in events] == ['start', 'stop']
