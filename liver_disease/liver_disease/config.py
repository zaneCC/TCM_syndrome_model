import constants


IS_SHE = True

''' 症状-证
汇总表-未做特征选择:    MERGE_CSV_PATH
汇总表-特征选择后:     SELECTION_MERGE_CSV_PATH
人工选择特征:         SYMP_MAIN_ACC_DIAGNOSIS_PATH
随机过采样:           RANDOM_OVER_SAMPLER_CSV_PATH
SMOTE:              SMOTE_MERGE_CSV_PATH
SMOTE Borderline1:  SMOTE_BORDERLINE1_MERGE_CSV_PATH
SMOTE_D:            SMOTE_D_MERGE_CSV_PATH
SMOTE_BORDERLINE_D: SMOTE_Borderline_D_CSV_PATH
'''
''' 症状-舌象-证
汇总表-舌象:          MERGE_CSV_DIA_TONGUE_PATH
随机过采样:           TUE_RANDOM_OVER_SAMPLER_CSV_PATH
SMOTE:              TUE_SMOTE_MERGE_CSV_PATH
SMOTE Borderline1:  TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH
SMOTE_D:            TUE_SMOTE_D_MERGE_CSV_PATH
# SMOTE_BORDERLINE_D: TUE_SMOTE_Borderline_D_CSV_PATH
'''
PATH = constants.TUE_SMOTE_MERGE_CSV_PATH

def config_path_compare_test(j):
    # RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + '实验' + '/对比实验/' + str(
    #     j) + '/随机过采样-采样-汇总表.csv'
    # SMOTE_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE-采样-汇总表.csv'
    # SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE_Borderline1-采样-汇总表.csv'
    # SMOTE_D_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE_D-采样-汇总表.csv'
    # SMOTE_Borderline_D_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE_Borderline_D-采样-汇总表.csv'

    ''' 修改论文数据的时候使用 '''
    RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/对比实验/' + str(
        j) + '/随机过采样-采样-汇总表.csv'
    SMOTE_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/对比实验/' + str(
        j) + '/SMOTE-采样-汇总表.csv'
    SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/对比实验/' + str(
        j) + '/SMOTE_Borderline1-采样-汇总表.csv'
    SMOTE_D_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/对比实验/' + str(
        j) + '/SMOTE_D-采样-汇总表.csv'
    SMOTE_Borderline_D_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/对比实验/' + str(
        j) + '/SMOTE_Borderline_D-采样-汇总表.csv'

    # 舌象-证，数据
    # if config.IS_SHE:
    #     RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/舌象/对比实验/' + str(
    #         j) + '/随机过采样-采样-汇总表.csv'
    #     SMOTE_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/舌象/对比实验/' + str(
    #         j) + '/SMOTE-采样-汇总表.csv'
    #     SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/舌象/对比实验/' + str(
    #         j) + '/SMOTE_Borderline1-采样-汇总表.csv'
    #     SMOTE_D_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/舌象/对比实验/' + str(
    #         j) + '/SMOTE_D-采样-汇总表.csv'
    #     SMOTE_Borderline_D_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/舌象/对比实验/' + str(
    #         j) + '/SMOTE_Borderline_D-采样-汇总表.csv'

    # 舌象-证，数据
    ''' 修改论文数据的时候使用 '''
    if IS_SHE:
        RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/症状_舌象/对比实验/' + str(
            j) + '/随机过采样-采样-汇总表.csv'
        SMOTE_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/症状_舌象/对比实验/' + str(
            j) + '/SMOTE-采样-汇总表.csv'
        SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/症状_舌象/对比实验/' + str(
            j) + '/SMOTE_Borderline1-采样-汇总表.csv'
        SMOTE_D_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/症状_舌象/对比实验/' + str(
            j) + '/SMOTE_D-采样-汇总表.csv'
        SMOTE_Borderline_D_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/症状_舌象/对比实验/' + str(
            j) + '/SMOTE_Borderline_D-采样-汇总表.csv'
    path = SMOTE_MERGE_COMPARE_CSV_PATH
    return path