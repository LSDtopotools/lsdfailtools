'''
from plot_iverson_lsdtt.sensitivity_analysis import MonteCarloIverson

my_simulation = MonteCarloIverson(range_D_0 = [1e-8,1e-4], range_Ksat = [1e-11,1e-6],range_Iz_over_K_steady = [0.1,1.5], range_d = [0.5,3], range_alpha = [0.1,1.5], range_friction_angle = [0.1,1],
        range_cohesion = [100,5000], range_weight_of_soil = [15000,30000], range_weight_of_water = [9800,9800], depth_spacing = 0.1, n_depths = 35, OMS_D_0 = 1.,OMS_Ksat = 1., OMS_d = 0.1 , OMS_alpha = 0.1, OMS_friction_angle = 0.05, OMS_cohesion = 100,OMS_weight_of_soil = 500,OMS_weight_of_water = 100, OMS_Iz_over_Kz = 0.1,
        program = "", path_to_rainfall_csv = "./", rainfall_csv = "preprocessed_data.csv", path_to_root_analysis = "./", suffix = 0)



my_simulation.run_MonteCarlo(failure_time_s = 540460, n_proc = 4, n_tests = 1000)

'''
from plot_iverson_lsdtt.sensitivity_analysis import MonteCarloIverson

program = "./cpp_model/Analysis_driver/Iverson_Model.exe"
MySim = MonteCarloIverson(program = program, path_to_rainfall_csv = "./test_data/", rainfall_csv = "preprocessed_data.csv", path_to_root_analysis = "./test_run_marina_output/", depth_spacing = 0.1,
	n_depths = 32, range_D_0 = [5e-7,5e-5], range_Ksat = [5e-10,5e-7],range_Iz_over_K_steady = [0.1,0.8], range_d = [1,3], range_alpha = [0.5,1.1], range_friction_angle = [0.3,0.5],
        range_cohesion = [15500,16500], range_weight_of_soil = [15000,20000], range_weight_of_water = [9800,9800],
        OMS_D_0 = 0.05,OMS_Ksat = 0.05, OMS_d = 0.05, OMS_alpha = 0.05, OMS_friction_angle = 0.03, OMS_cohesion = 100,OMS_weight_of_soil = 100,OMS_weight_of_water = 100, OMS_Iz_over_Kz = 0.05)

MySim.run_MonteCarlo(failure_time_s = 540460, n_proc = 4, n_tests = 1000)
