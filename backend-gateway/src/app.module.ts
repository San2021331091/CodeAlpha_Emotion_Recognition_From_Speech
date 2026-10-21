import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { SpeechEmotionModule } from './speech-emotion/speech-emotion.module';
import { EmotionHistoryModule } from './emotion-history/emotion-history.module';
import { AnalyticsModule } from './analytics/analytics.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
    SpeechEmotionModule,
    EmotionHistoryModule,
    AnalyticsModule,
  ],
})
export class AppModule {}
