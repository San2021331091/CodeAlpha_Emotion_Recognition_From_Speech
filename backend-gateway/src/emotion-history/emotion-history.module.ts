import { Module } from '@nestjs/common';
import { EmotionHistoryService } from './emotion-history.service';
import { EmotionHistoryController } from './emotion-history.controller';

@Module({
  controllers: [EmotionHistoryController],
  providers: [EmotionHistoryService],
  exports: [EmotionHistoryService],
})
export class EmotionHistoryModule {}
