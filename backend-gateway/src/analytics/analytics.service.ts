import { Injectable } from '@nestjs/common';
import { EmotionHistoryService } from '../emotion-history/emotion-history.service';

@Injectable()
export class AnalyticsService {
  constructor(private readonly historyService: EmotionHistoryService) {}

  getSummary() {
    const records = this.historyService.findAll();
    const totalUtterances = records.length;

    const emotionCounts: Record<string, number> = {
      neutral: 0,
      calm: 0,
      happy: 0,
      sad: 0,
      angry: 0,
      fearful: 0,
      disgust: 0,
      surprised: 0,
    };

    let totalConfidence = 0;
    const confidenceByEmotion: Record<string, { sum: number; count: number }> = {};
    let agreements = 0;

    for (const r of records) {
      const emo = (r.predictedEmotion || 'neutral').toLowerCase();
      emotionCounts[emo] = (emotionCounts[emo] || 0) + 1;

      totalConfidence += r.confidence;
      if (!confidenceByEmotion[emo]) {
        confidenceByEmotion[emo] = { sum: 0, count: 0 };
      }
      confidenceByEmotion[emo].sum += r.confidence;
      confidenceByEmotion[emo].count += 1;

      if (r.verifiedEmotion && r.verifiedEmotion.toLowerCase() === emo) {
        agreements += 1;
      }
    }

    const avgConfidenceByEmotion: Record<string, number> = {};
    for (const [key, val] of Object.entries(confidenceByEmotion)) {
      avgConfidenceByEmotion[key] = Number((val.sum / val.count).toFixed(3));
    }

    return {
      totalUtterances,
      averageConfidence: totalUtterances > 0 ? Number((totalConfidence / totalUtterances).toFixed(3)) : 0.0,
      userAgreementRate: totalUtterances > 0 ? Number(((agreements / totalUtterances) * 100).toFixed(1)) : 100.0,
      emotionDistribution: emotionCounts,
      averageConfidenceByEmotion: avgConfidenceByEmotion,
      recentActivity: records.slice(0, 5),
    };
  }
}
