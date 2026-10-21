import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  Query,
  Res,
  HttpStatus,
} from '@nestjs/common';
import { Response } from 'express';
import { EmotionHistoryService } from './emotion-history.service';
import { CreateHistoryEntryDto, UpdateHistoryEntryDto } from './dto/create-history-entry.dto';

@Controller('emotion-history')
export class EmotionHistoryController {
  constructor(private readonly historyService: EmotionHistoryService) {}

  @Post()
  create(@Body() dto: CreateHistoryEntryDto) {
    return this.historyService.create(dto);
  }

  @Get()
  findAll(@Query('emotion') emotion?: string, @Query('search') search?: string) {
    return this.historyService.findAll(emotion, search);
  }

  @Get('export/csv')
  exportCsv(@Res() res: Response) {
    const csvData = this.historyService.exportCsv();
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename="speech_emotion_history.csv"');
    res.status(HttpStatus.OK).send(csvData);
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.historyService.findOne(id);
  }

  @Patch(':id')
  update(@Param('id') id: string, @Body() dto: UpdateHistoryEntryDto) {
    return this.historyService.update(id, dto);
  }

  @Delete(':id')
  remove(@Param('id') id: string) {
    return this.historyService.remove(id);
  }
}
