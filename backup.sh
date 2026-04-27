#!/bin/bash
BACKUP_DIR="/Users/valenwei/jobtrack/backups"
DATE=$(date +%Y%m%d_%H%M%S)
FILE="$BACKUP_DIR/jobsdb_$DATE.sql.gz"

docker exec docker-db_postgres-1 pg_dump -U postgres jobsdb | gzip > "$FILE"

# 只保留最近 7 天的备份
find "$BACKUP_DIR" -name "jobsdb_*.sql.gz" -mtime +7 -delete

echo "[$(date)] Backup saved: $FILE ($(du -sh $FILE | cut -f1))"
