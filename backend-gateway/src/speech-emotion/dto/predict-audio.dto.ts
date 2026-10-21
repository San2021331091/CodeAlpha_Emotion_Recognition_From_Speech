import { IsOptional, IsString, MaxLength } from 'class-validator';

export class PredictAudioDto {
  @IsOptional()
  @IsString()
  modelName?: string;

  @IsOptional()
  @IsString()
  @MaxLength(64)
  requestId?: string;
}

export interface ValenceArousalDto {
  valence: number;
  arousal: number;
  quadrant: string;
}

export interface AcousticSummaryDto {
  duration_seconds: number;
  sample_rate: number;
  rms_energy_mean: number;
  zero_crossing_rate_mean: number;
  spectral_centroid_mean: number;
  spectral_rolloff_mean: number;
  pitch_f0_mean: number;
  pitch_f0_std: number;
  tempo: number;
  mfcc_means: number[];
  chroma_means: number[];
}

export interface EmotionPredictionDto {
  request_id?: string;
  input_sha256?: string;
  top_emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
  valence_arousal: ValenceArousalDto;
  acoustic_summary: AcousticSummaryDto;
  model_used: string;
  inference_time_ms: number;
  explanation: string;
}
