"""The core's door to the native runtime and the engines.

- ``service.InferenceService``: the runtime (devices, the model manager), built lazily, and what is installed
  (an engine's missing packages, its optional extras, its trainer).
- ``runtime.RuntimeService``: ``loaded(model, device, …)`` gives a model's ``EngineHandle``; behind it one engine
  per (engine, device), here or on a remote host, the models they hold, unloading and the doctor.
- ``engine``: ``EngineHandle`` (what ``TtsService`` and ``AsrService`` call: ``prepare_voice``, ``synthesize``,
  ``convert``, ``edit``, ``call``) and ``EngineRequest`` (cancel, timeout, streamed text, progress; what the engine
  reported).
- ``local``: ``LocalEngine`` and ``LocalHandle``, an engine and its models in this process.
- ``tasks``: the request and result types engines take and give (standard library only).

Only this package imports ``mizuoto.runtime``, ``mizuoto.engines`` and ``mizuoto.training``, inside functions.
"""
