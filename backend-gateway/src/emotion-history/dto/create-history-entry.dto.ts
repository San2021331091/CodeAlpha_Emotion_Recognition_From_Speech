import { IsString, IsNumber, IsOptional, IsObject } from 'class-validator';

export class CreateHistoryEntryDto {
  @IsString()
  title: string;

  @IsString()
  predictedEmotion: string;

  @IsNumber()
  confidence: number;

  @IsString()
  modelUsed: string;

  @IsOptional()
  @IsString()
  verifiedEmotion?: string;

  @IsOptional()
  @IsString()
  notes?: string;

  @IsOptional()
  @IsObject()
  acousticSummary?: any;

  @IsOptional()
  @IsObject()
  valenceArousal?: any;
}

export class UpdateHistoryEntryDto {
  @IsOptional()
  @IsString()
  verifiedEmotion?: string;

  @IsOptional()
  @IsString()
  notes?: string;

  @IsOptional()
  @IsString()
  title?: string;
}

export interface EmotionHistoryRecord {
  id: string;
  title: string;
  predictedEmotion: string;
  confidence: number;
  modelUsed: string;
  verifiedEmotion?: string;
  notes?: string;
  acousticSummary?: any;
  valenceArousal?: any;
  createdAt: string;
}
