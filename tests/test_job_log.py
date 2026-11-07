from test_workbench import client


def test_http_log_tail_is_bounded_and_preserves_recent_utf8(client, tmp_path, monkeypatch):
    from hi3d_workbench import jobs, server, common
    monkeypatch.setattr(jobs, 'DATA', tmp_path / 'data')
    manager = jobs.JobManager(); monkeypatch.setattr(server, 'manager', manager)
    folder = manager.root / 'log-fixture'; folder.mkdir()
    common.atomic_json(folder / 'state.json', {'id': folder.name, 'created': 1,
        'state': 'running', 'options': {}})
    end = '\n中文阶段进度 99 / 100\n'.encode('utf-8')
    (folder / 'worker.log').write_bytes(b'OLD CONTENT\n' + b'x' * (8 * 1024 * 1024) + end)
    response = client.get('/api/jobs/log-fixture/log?limit=4096')
    assert response.status_code == 200 and len(response.content) < 4300
    assert '中文阶段进度 99 / 100' in response.text and 'OLD CONTENT' not in response.text
    assert response.headers['x-log-truncated'] == 'true'
    assert int(response.headers['x-log-size']) > 8 * 1024 * 1024
    assert response.headers['cache-control'] == 'no-store'
    (folder / 'worker.log').write_bytes(end)
    response = client.get('/api/jobs/log-fixture/log')
    assert response.status_code == 200 and response.content == end
    assert response.headers['x-log-truncated'] == 'false'
    assert client.get('/api/jobs/log-fixture/log?limit=999999999').status_code == 422
    assert client.get('/api/jobs/missing/log').status_code == 404
