import {
  Controller,
  Post,
  Get,
  Param,
  Body,
  UseInterceptors,
  UploadedFile,
  Res,
  HttpStatus,
  Query,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { Response } from 'express';
import { SpeechEmotionService } from './speech-emotion.service';
import { PredictAudioDto } from './dto/predict-audio.dto';

@Controller('speech-emotion')
export class SpeechEmotionController {
  constructor(private readonly speechEmotionService: SpeechEmotionService) {}
  @Post('predict')
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 25 * 1024 * 1024 } }))
  async predictAudio(
    @UploadedFile() file: any,
    @Body() dto: PredictAudioDto,
  ) {
    return await this.speechEmotionService.predictAudio(file, dto.modelName, dto.requestId);
  }

  @Get('metrics')
  async getMetrics() {
    return await this.speechEmotionService.getMetrics();
  }

  @Get('samples')
  async getSamples() {
    return await this.speechEmotionService.getSamples();
  }

  @Get('sample/:id')
  async streamSample(@Param('id') id: string, @Res() res: Response) {
    const { buffer, contentType } = await this.speechEmotionService.getSampleAudioBuffer(id);
    res.setHeader('Content-Type', contentType);
    res.setHeader('Content-Length', buffer.length);
    res.status(HttpStatus.OK).send(buffer);
  }

  @Post('retrain')
  async triggerRetrain(@Query('samples_per_emotion') count?: number) {
    return await this.speechEmotionService.triggerRetraining(count);
  }

  @Get('retrain/status')
  async getTrainingStatus() {
    return await this.speechEmotionService.getTrainingStatus();
  }
}
