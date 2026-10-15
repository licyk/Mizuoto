import type { components } from '@/api/schema';

/** Shorthand for the generated schemas (one per model: separate_input_output_schemas=False). */
export type S = components['schemas'];

export type AppMeta = S['AppMeta'];
export type SettingsView = S['SettingsView'];
export type Job = S['Job'];
export type JobStep = S['JobStep'];
export type JobPage = S['JobPage'];
export type JobLog = S['JobLog'];
export type Output = S['Output'];
export type OutputPage = S['OutputPage'];
export type AudioFile = S['AudioFile'];
export type Peaks = S['Peaks'];
export type BrowseListing = S['BrowseListing'];
export type BrowseEntry = S['BrowseEntry'];
export type ComputeUsage = S['ComputeUsage'];
export type ComputeInfo = S['ComputeInfo'];
export type DeviceInfo = S['DeviceInfo'];
export type RuntimeStatus = S['RuntimeStatus'];
export type LoadedModelInfo = S['LoadedModelInfo'];
export type EngineRuntime = S['EngineRuntime'];

export type EngineStatus = S['EngineStatus'];
export type EngineDetail = S['EngineDetail'];
export type EngineManifest = S['EngineManifest'];
export type EngineSettings = S['EngineSettings'];
export type DoctorFinding = S['DoctorFinding'];
export type DoctorReport = S['DoctorReport'];
export type ModelInfo = S['ModelInfo'];
export type AssetInfo = S['AssetInfo'];
export type ModelParamSpecs = S['ModelParamSpecs'];
export type ParamField = S['ParamField'];
export type VariantInfo = S['VariantInfo'];

export type Voice = S['Voice'];
export type VoiceSummary = S['VoiceSummary'];
export type VoiceReference = S['VoiceReference'];
export type VoiceFit = S['VoiceFit'];
export type Transcript = S['Transcript'];

// The request models are also a response (an output's re-run), so FastAPI splits them: send -Input, read -Output.
export type SpeakRequest = S['SpeakRequest-Input'];
export type SpeakRequestOut = S['SpeakRequest-Output'];
export type CommonParams = S['CommonParams-Input'];
export type SynthesisPlan = S['SynthesisPlan'];
export type SegmentPreview = S['SegmentPreview'];
export type VoiceInput = S['VoiceInput-Input'];
export type EmotionInput = S['EmotionInput-Input'];
export type Preset = S['Preset'];

export type Dataset = S['Dataset'];
export type DatasetRow = S['DatasetRow'];
export type DatasetReport = S['DatasetReport'];
export type PrepareRequest = S['PrepareRequest'];
export type Experiment = S['Experiment'];
export type StageState = S['StageState'];
export type MethodInfo = S['MethodInfo'];
export type TrainingMethod = S['TrainingMethod'];
export type OptionSpec = S['OptionSpec'];
export type MetricPoint = S['MetricPoint'];
export type PresetInput = S['PresetInput'];

/** Socket event payloads, typed from the same schema as REST. */
export type ServerEvents = { [K in keyof S['ServerEvents']]: S['ServerEvents'][K] };
export type JobState = Job['state'];
export type VoiceMode = ModelParamSpecs['voice_modes'][number];
export type Capability = ModelParamSpecs['capabilities'][number];

export type Project = S['Project'];
export type ProjectSummary = S['ProjectSummary'];
export type ProjectInput = S['ProjectInput'];
export type ProjectLine = S['Line-Output'];
export type ProjectChapter = S['Chapter-Output'];
export type ProjectSpeaker = S['Speaker-Output'];
export type LineInput = S['Line-Input'];
export type ChapterInput = S['Chapter-Input'];
export type SpeakerInput = S['Speaker-Input'];
