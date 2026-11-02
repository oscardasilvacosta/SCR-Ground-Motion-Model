# Fixed diagnostic fits: mixed recording validation or training-target permutation.
script <- sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(normalizePath(script)),'common.R'))
args <- commandArgs(TRUE)
if(length(args)!=3L || !args[1]%in%c('mixed','permutation'))
  stop('Usage: Rscript R/07_diagnostics.R mixed|permutation PRIVATE_training.csv NEW_output_directory')
d <- read_training(args[2]);out <- new_output(args[3]);require_model_packages();rows <- list()
if(args[1]=='mixed') {
  set.seed(123);train <- caret::createDataPartition(d[[target]],p=.7,list=FALSE)[,1]
  test <- setdiff(seq_len(nrow(d)),train)
  model <- fit_model(d[train,],seed=365209L,evaluation_final_seed=365209L)
  p <- as.numeric(predict(model,newdata=d[test,]))
  write.csv(score(d[[target]][test],p),file.path(out,'mixed_metrics.csv'),row.names=FALSE)
  write.csv(data.frame(record_id=d$record_id[test],observed_pctg=d[[target]][test],predicted_pctg=p),
    file.path(out,'mixed_predictions.csv'),row.names=FALSE)
} else {
  split <- primary_partition()
  for(r in seq_len(20L)) {
    training <- d[split$train,];seed <- 2026102000L+r;set.seed(seed)
    training[[target]] <- sample(training[[target]]);model <- fit_model(training,seed=seed)
    p <- as.numeric(predict(model,newdata=d[split$test,]))
    rows[[r]] <- data.frame(repetition=r,seed=seed,score(d[[target]][split$test],p))
    write.csv(do.call(rbind,rows),file.path(out,'permutation_20_metrics.csv'),row.names=FALSE)
    rm(model);gc(verbose=FALSE)
  }
}
