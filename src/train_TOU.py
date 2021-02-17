from TOU_analysis_and_prediction import TOU

tou_file = 'full_data.csv'
tou_subdir = 'data/TOU_Data'

tou_obj = TOU(tou_file, tou_subdir)
tou_obj.create_and_fit_model(seasonality = 12, fitted_model_filename='cluster_TOU.pickle')

# with open('test_file.txt', 'w') as f:
#     f.write("This is a sample file")
