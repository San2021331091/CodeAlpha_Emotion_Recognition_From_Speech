import { Module } from '@nestjs/common';
import { SpeechEmotionController } from './speech-emotion.controller';
import { SpeechEmotionService } from './speech-emotion.service';

@Module({
  controllers: [SpeechEmotionController],
  providers: [SpeechEmotionService],
  exports: [SpeechEmotionService],
})
export class SpeechEmotionModule {}
