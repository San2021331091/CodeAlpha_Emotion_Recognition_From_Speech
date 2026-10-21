import { Injectable, NotFoundException } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid';
import { CreateHistoryEntryDto, UpdateHistoryEntryDto, EmotionHistoryRecord } from './dto/create-history-entry.dto';

@Injectable()
export class EmotionHistoryService {
  private history: EmotionHistoryRecord[] = [];

  create(dto: CreateHistoryEntryDto): EmotionHistoryRecord {
    const record: EmotionHistoryRecord = {
      id: uuidv4(),
      title: dto.title || 'Untitled Speech Analysis',
      predictedEmotion: dto.predictedEmotion,
      confidence: dto.confidence,
      modelUsed: dto.modelUsed,
      verifiedEmotion: dto.verifiedEmotion || dto.predictedEmotion,
      notes: dto.notes || '',
      acousticSummary: dto.acousticSummary || null,
      valenceArousal: dto.valenceArousal || null,
      createdAt: new Date().toISOString(),
    };

    this.history.unshift(record);
    return record;
  }

  findAll(emotionFilter?: string, search?: string): EmotionHistoryRecord[] {
    let list = this.history;
    if (emotionFilter && emotionFilter !== 'all') {
      list = list.filter(
        (item) =>
          item.predictedEmotion.toLowerCase() === emotionFilter.toLowerCase() ||
          (item.verifiedEmotion && item.verifiedEmotion.toLowerCase() === emotionFilter.toLowerCase()),
      );
    }
    if (search && search.trim() !== '') {
      const q = search.toLowerCase();
      list = list.filter(
        (item) => item.title.toLowerCase().includes(q) || (item.notes && item.notes.toLowerCase().includes(q)),
      );
    }
    return list;
  }

  findOne(id: string): EmotionHistoryRecord {
    const item = this.history.find((h) => h.id === id);
    if (!item) {
      throw new NotFoundException(`History record with ID ${id} not found.`);
    }
    return item;
  }

  update(id: string, dto: UpdateHistoryEntryDto): EmotionHistoryRecord {
    const item = this.findOne(id);
    if (dto.verifiedEmotion !== undefined) item.verifiedEmotion = dto.verifiedEmotion;
    if (dto.notes !== undefined) item.notes = dto.notes;
    if (dto.title !== undefined) item.title = dto.title;
    return item;
  }

  remove(id: string): { success: boolean; id: string } {
    const index = this.history.findIndex((h) => h.id === id);
    if (index === -1) {
      throw new NotFoundException(`History record with ID ${id} not found.`);
    }
    this.history.splice(index, 1);
    return { success: true, id };
  }

  exportCsv(): string {
    const headers = [
      'ID',
      'Title',
      'Predicted Emotion',
      'Confidence',
      'Verified Emotion',
      'Model Used',
      'Duration (s)',
      'Pitch F0 (Hz)',
      'RMS Energy',
      'Created At',
      'Notes',
    ];

    const rows = this.history.map((h) => [
      h.id,
      `"${(h.title || '').replace(/"/g, '""')}"`,
      h.predictedEmotion,
      (h.confidence * 100).toFixed(1) + '%',
      h.verifiedEmotion || h.predictedEmotion,
      `"${h.modelUsed}"`,
      h.acousticSummary?.duration_seconds ?? 'N/A',
      h.acousticSummary?.pitch_f0_mean ?? 'N/A',
      h.acousticSummary?.rms_energy_mean ?? 'N/A',
      h.createdAt,
      `"${(h.notes || '').replace(/"/g, '""')}"`,
    ]);

    return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
  }
}
