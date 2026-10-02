import numpy as np
import solcx
class ParameterConfig(object):
    """
     Some Configurations
    """
    FF = './'
    viewfile = FF+'/dataset/TP/'
    GAT_HIDDEN_DIM = 128
    HEAD_NUM = 5
    GAT_Layer_Num = 1
    GAT_FEAT_DP_RATE = 0.
    GAT_ATT_DP_RATE = 0.
    device = None
    EMBEDDING_DIM = 100
    MAX_SEQUENCE_LENGTH = 20
    EPOCHES = 200
    BATCH_SIZE = 128
    dataset_split_ratio = 0.2
    OCCUPY_ALL = False
    PRINT_PER_BATCH = 100
    PRE_TRAINING = True
    SEED = np.random.seed()
    lr = 1e-3
    lr_decay = 0.9
    lr_decay = 0.1
    clip = 1.0
    l2_reg_lambda = 0.01
    NUM_FILTERS = 128
    FILTER_SIZES = [2, 3, 4]
    DROP_OUT = 0.5
    CFG_MIN_EDGE_NUM = 1
    PIN_MEM = False
    NUM_WORKERS = 4
    solcx.install_solc('v0.4.25')
    solcx.set_solc_version('v0.4.25')
    ROOT = Path(__file__).resolve().parent
    viewfile = str(ROOT / 'artifacts')
    GAT_HIDDEN_DIM=128; HEAD_NUM=5; GAT_Layer_Num=1
    GAT_FEAT_DP_RATE=0.; GAT_ATT_DP_RATE=0.; EMBEDDING_DIM=100
    EPOCHES=200; BATCH_SIZE=128; dataset_split_ratio=.2; lr=1e-3
    SEED=123; CFG_MIN_EDGE_NUM=1; PIN_MEM=False; NUM_WORKERS=0
    def log_config(prefix):
        with open(prefix + '#config', 'w') as f:
            f.write('EMBEDDING_DIM =' + str(ParameterConfig.EMBEDDING_DIM) + '\n')
            f.write('MAX_SEQUENCE_LENGTH=' + str(ParameterConfig.MAX_SEQUENCE_LENGTH) + '\n')
            f.write('GCN_HIDDEN_DIM='+str(ParameterConfig.GCN_HIDDEN_DIM) + '\n')
            f.write('GAT_HIDDEN_DIM='+str(ParameterConfig.GAT_HIDDEN_DIM) + '\n')
            f.write('HEAD_NUM='+str(ParameterConfig.HEAD_NUM) + '\n')
            f.write('CFG_MIN_EDGE_NUM='+str(ParameterConfig.CFG_MIN_EDGE_NUM) + '\n')
            f.write('NUM_FILTERS=' + str(ParameterConfig.NUM_FILTERS) + '\n')
            f.write('FILTER_SIZES=' + str(ParameterConfig.FILTER_SIZES) + '\n')
            f.write('DROP_OUT=' + str(ParameterConfig.DROP_OUT) + '\n')
            f.write('BATCH_SIZE=' + str(ParameterConfig.BATCH_SIZE) + '\n')
            f.write('EPOCHES=' + str(ParameterConfig.EPOCHES) + '\n')
            f.write('LEARNING_RATE=' + str(ParameterConfig.lr) + '\n')
            f.write('DATASET_SPLIT_RATIO=' + str(ParameterConfig.dataset_split_ratio) + '\n')
            f.write('HIDDEN_DIM=' + str(ParameterConfig.HEAD_NUM) + '\n')
            f.write('HEAD_NUM=' + str(ParameterConfig.HEAD_NUM) + '\n')
            f.close()
