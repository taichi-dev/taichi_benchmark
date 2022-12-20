cuda_sample_results = {'cuda_2d': [{'n_particles': 512, 'time_ms': 0.306909}, {'n_particles': 2048, 'time_ms': 0.328294}, {'n_particles': 4608, 'time_ms': 0.411246}, {'n_particles': 8192, 'time_ms': 0.506373}, {'n_particles': 12800, 'time_ms': 0.631633}, {'n_particles': 18432, 'time_ms': 0.74418}, {'n_particles': 25088, 'time_ms': 0.889422}, {'n_particles': 32768, 'time_ms': 1.087441}], 'cuda_3d': [{'n_particles': 8192, 'time_ms': 0.941462}, {'n_particles': 65536, 'time_ms': 5.481499}, {'n_particles': 221184, 'time_ms': 21.760391}, {'n_particles': 524288, 'time_ms': 99.960159}, {'n_particles': 1024000, 'time_ms': 343.911346}, {'n_particles': 1769472, 'time_ms': 831.018188}, {'n_particles': 2809856, 'time_ms': 1670.747681}, {'n_particles': 4194304, 'time_ms': 2585.545654}]}

taichi_sample_results = {
    'taichi_2d': [{
        'n_particles': 512,
        'time_ms': 0.4708201557548364
    }, {
        'n_particles': 2048,
        'time_ms': 0.4732567070391269
    }, {
        'n_particles': 4608,
        'time_ms': 0.5115689839101378
    }, {
        'n_particles': 8192,
        'time_ms': 0.5668949448249805
    }, {
        'n_particles': 12800,
        'time_ms': 0.6931407124000089
    }, {
        'n_particles': 18432,
        'time_ms': 0.8494079414163025
    }, {
        'n_particles': 25088,
        'time_ms': 1.0304002558427783
    }, {
        'n_particles': 32768,
        'time_ms': 1.2111213139576193
    }],
    'taichi_3d': [{
        'n_particles': 8192,
        'time_ms': 1.9398335703044722
    }, {
        'n_particles': 65536,
        'time_ms': 10.701219556153774
    }, {
        'n_particles': 221184,
        'time_ms': 34.867725920889825
    }, {
        'n_particles': 524288,
        'time_ms': 140.94088707861374
    }, {
        'n_particles': 1024000,
        'time_ms': 549.7929778569244
    }, {
        'n_particles': 1769472,
        'time_ms': 1286.6532287104633
    }, {
        'n_particles': 2809856,
        'time_ms': 2374.8291689565235
    }, {
        'n_particles': 4194304,
        'time_ms': 4005.1647448022436
    }]
}

