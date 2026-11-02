# Requires the authors' source workbook, not an unmodified ESM download.
script <- sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(normalizePath(script)),'common.R'))
args <- commandArgs(TRUE)
if(length(args)!=2L) stop('Usage: Rscript R/01_prepare_data.R PRIVATE_source.xlsx NEW_PRIVATE_output_directory')
if(!requireNamespace('readxl',quietly=TRUE)) stop('Install readxl for source-workbook preparation.')
raw <- as.data.frame(readxl::read_excel(args[1]))
required <- c(target,'Mw','event_depth_km','epicentral_distance_km','hypocentral_distance_km',
  'vs30_m_s','azimuth_cos','azimuth_sin','event_latitude','event_longitude','station_latitude','station_longitude')
missing <- setdiff(required,names(raw))
if(length(missing)) stop('Source workbook is missing: ',paste(missing,collapse=', '))
eligible <- complete.cases(raw[required]);d <- raw[eligible,]
d$source_Excel_row <- which(eligible)+1L;d$record_id <- seq_len(nrow(d))
stopifnot(nrow(d)==9579L)
# Historical angular columns define eligibility; recreate the adopted angular feature.
d <- geometry(d)
keep <- unique(c('record_id','source_Excel_row',target,features,'station_longitude'))
out <- new_output(args[2]);path <- file.path(out,'training_database.csv')
write.csv(d[keep],path,row.names=FALSE)
invisible(read_training(path))
cat('Prepared 9,579 records. Original distances, Vs30 and raw %g target retained. Keep this output private.\n')
