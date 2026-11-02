# Exact TreeSHAP; per-record outputs remain local, not automatically published.
script <- sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(normalizePath(script)),'common.R'))
args <- commandArgs(TRUE)
if(length(args)!=3L) stop('Usage: Rscript R/08_shap.R model.rds PRIVATE_inputs.csv NEW_output_directory')
require_model_packages();model <- readRDS(args[1]);stopifnot(identical(model$coefnames,features))
d <- read.csv(args[2],check.names=FALSE);check_numeric(d,features);out <- new_output(args[3])
x <- as.matrix(predict(model$preProcess,newdata=d[features]))
shap <- predict(model$finalModel,x,predcontrib=TRUE,approxcontrib=FALSE)
p <- as.numeric(predict(model,newdata=d));stopifnot(max(abs(rowSums(shap)-p))<1e-3)
write.csv(data.frame(Feature=features,mean_absolute_SHAP_pctg=colMeans(abs(shap[,features,drop=FALSE]))),
  file.path(out,'SHAP_importance.csv'),row.names=FALSE)
write.csv(shap,file.path(out,'SHAP_values.csv'),row.names=FALSE)
cat('SHAP is in %g. Descriptive attributions are not causal effects.\n')
