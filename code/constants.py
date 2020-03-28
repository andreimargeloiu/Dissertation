LOGS_TABLE = 'logs'

LOGS_SCHEMA = {
    'epoch':int,
    'train_loss':float,
    'valid_loss':float,
    'train_error_rate':float,
    'valid_error_rate':float,
    'valid_adv_error_rate':float,
    'time':float
}

ARGS_TABLE = 'metadata'
ARGS_SCHEMA = {
    'epochs':int,
    'batch_size':int,
    'model':str,
    'time':str
}