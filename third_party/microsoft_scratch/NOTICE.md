Microsoft Bringing Old Photos Back to Life, MIT license.
networks.py: relative antialiasing import; DataParallel alias replaces unused sync-BN import.
Inference uses sync_bn=False on CPU; no network layers or weight keys changed.
