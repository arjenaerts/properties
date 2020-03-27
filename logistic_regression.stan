data {
    int<lower=1> n_rows_train;
    int<lower=1> n_rows_test;
    int<lower=1> n_variables;
    matrix[n_rows_train, n_variables] X_train;
    matrix[n_rows_test, n_variables] X_test;
    int<lower=0, upper=1> y_train[n_rows_train];
}

parameters{
   vector[n_variables] beta;
}

model{
   beta ~ normal(0,1);    
   y_train ~ bernoulli(inv_logit(X_train * beta));
}

generated quantities {
    int<lower=0, upper=1> y_pred_train[n_rows_train]; 
    int<lower=0, upper=1> y_pred_test[n_rows_test];
    
    y_pred_train = bernoulli_rng(inv_logit(X_train * beta));
    y_pred_test = bernoulli_rng(inv_logit(X_test * beta));
}
