import { Injectable, HttpException, HttpStatus, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { EmotionPredictionDto } from './dto/predict-audio.dto';

@Injectable()
export class SpeechEmotionService {
  private readonly logger = new Logger(SpeechEmotionService.name);
  private readonly mlBaseUrl: string;
  constructor(private readonly configService: ConfigService) {
    this.mlBaseUrl = this.configService.get<string>('ML_SERVICE_URL');
    this.logger.log(`Initialized SpeechEmotionService with ML URL: ${this.mlBaseUrl}`);
  }
  async predictAudio(file: any, modelName?: string, requestId?: string): Promise<EmotionPredictionDto> {
    if (!file || !file.buffer || file.buffer.length === 0) {
      throw new HttpException('No audio file provided', HttpStatus.BAD_REQUEST);
    }

    try {
      const formData = new FormData();
      const blob = new Blob([file.buffer], { type: file.mimetype || 'audio/wav' });
      formData.append('file', blob, file.originalname || 'speech_sample.wav');
      if (requestId) formData.append('request_id', requestId);

      if (modelName) {
        formData.append('model_name', modelName);
      }

      const response = await fetch(`${this.mlBaseUrl}/api/predict/audio`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let message = "Audio analysis failed";
        try {
          const error = JSON.parse(errorText);
          if (typeof error.detail === "string") message = error.detail;
        } catch {}
        throw new HttpException(
          message,
          response.status || HttpStatus.INTERNAL_SERVER_ERROR,
        );
      }

      const data = (await response.json()) as EmotionPredictionDto;
      return data;
    } catch (err: any) {
      this.logger.error(`Failed to analyze audio in ML engine: ${err.message}`);
      if (err instanceof HttpException) throw err;
      throw new HttpException(`ML Gateway Service Unavailable: ${err.message}`, HttpStatus.BAD_GATEWAY);
    }
  }

  async getMetrics(): Promise<any> {
    try {
      const response = await fetch(`${this.mlBaseUrl}/api/models/metrics`);
      if (!response.ok) {
        throw new HttpException('Failed to fetch metrics from ML engine', response.status);
      }
      return await response.json();
    } catch (err: any) {
      this.logger.error(`Metrics fetch error: ${err.message}`);
      throw new HttpException(`Failed to retrieve ML metrics: ${err.message}`, HttpStatus.BAD_GATEWAY);
    }
  }

  async getSamples(): Promise<any[]> {
    try {
      const response = await fetch(`${this.mlBaseUrl}/api/audio/samples`);
      if (!response.ok) {
        return [];
      }
      return await response.json();
    } catch (err: any) {
      this.logger.warn(`Could not reach ML samples endpoint: ${err.message}`);
      return [];
    }
  }

  async getSampleAudioBuffer(sampleId: string): Promise<{ buffer: Buffer; contentType: string }> {
    try {
      const response = await fetch(`${this.mlBaseUrl}/api/audio/sample/${sampleId}`);
      if (!response.ok) {
        throw new HttpException(`Sample ${sampleId} not found`, HttpStatus.NOT_FOUND);
      }
      const arrayBuf = await response.arrayBuffer();
      const contentType = response.headers.get('content-type') || 'audio/wav';
      return {
        buffer: Buffer.from(arrayBuf),
        contentType,
      };
    } catch (err: any) {
      this.logger.error(`Stream audio sample error: ${err.message}`);
      if (err instanceof HttpException) throw err;
      throw new HttpException('Could not stream audio sample', HttpStatus.BAD_GATEWAY);
    }
  }

  async triggerRetraining(count?: number): Promise<any> {
  try {
    const url = new URL(`${this.mlBaseUrl}/api/train`);

    if (count !== undefined) {
      url.searchParams.set('samples_per_emotion', String(count));
    }

    const response = await fetch(url.toString(), {
      method: 'POST',
    });

    if (!response.ok) {
      const details = await response.json().catch(() => ({}));

      throw new HttpException(
        details.detail || 'Retraining failed on ML microservice',
        response.status,
      );
    }

    return await response.json();
  } catch (err: any) {
    if (err instanceof HttpException) {
      throw err;
    }

    throw new HttpException(
      `Retraining error: ${err.message}`,
      HttpStatus.BAD_GATEWAY,
    );
  }
}

  async getTrainingStatus(): Promise<any> {
    try {
      const response = await fetch(`${this.mlBaseUrl}/api/train/status`);
      if (!response.ok) throw new HttpException('Could not retrieve training status', response.status);
      return await response.json();
    } catch (err: any) {
      if (err instanceof HttpException) throw err;
      throw new HttpException(`Could not check training status: ${err.message}`, HttpStatus.BAD_GATEWAY);
    }
  }
}
