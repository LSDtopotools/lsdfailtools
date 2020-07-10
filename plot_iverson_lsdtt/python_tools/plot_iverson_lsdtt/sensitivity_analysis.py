# Used to manage the multiprocessing bit
import multiprocessing as mp
# Run the external code
import subprocess
# manages all the path issues 
import sys,glob,os
# Pandas is used for I/O table-like data (csv)
import pandas as pd
# Numpy for some numerical computations
import numpy as np
# Homemade code that deals with individual analysis plotting if needed
from plot_iverson_lsdtt import PLOTIV
# Well, sopy file copies files
from shutil import copyfile
# Deal with annoying int to str conversion that sometimes end up with like height zeros 
import decimal
# Basic log exp or stuff
import math
# Grant a random choice in list function
import random
# Manage sh commands from python
import shutil

########### Small function required for decimeal str debugging
# create a new context for this task
ctx = decimal.Context()
# 20 digits should be enough for everyone :D
ctx.prec = 20
def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

################################# Iversonisation ################################################################################
# Automation of sets of analysis from python

class Iversonisation(object):
    """
    Automate some analysis run to make easier sensitivity analysis.
    At the moment only work for the uni server as it requires the binary to be subproxcessed
    """
    def __init__(self, program, path_to_rainfall_csv = "./", rainfall_csv = "preprocessed_data.csv", path_to_root_analysis = "./"):
        """
        Initialise the Iversonisation, it needs to know where are the different components and the base analysis.
        Arguments:
            program (str): path and name of the cpp executable. At the moment test_Iverson_boris.out does the job.
            path_to_rainfall_csv (str): path to the common input of precipitations for this set of analysis. 
            rainfall_csv (str): name of the csv containing the precipitation intensity
            path_to_root_analysis(str): path to the root of the analysis, where a bunch of file will be created
        Returns:
            Iversonisation object
        Authors:
            B.G. - StackOverflow (for solving my multiprocess pool issues)
        Last update:
            12/02/2019
        """
        # get generic python object properties
        super(Iversonisation, self).__init__()
        # Different attributes explained in the docstring
        self.program = program
        self.path_to_rainfall_csv = path_to_rainfall_csv
        self.rainfall_csv = rainfall_csv
        self.path = path_to_root_analysis
        self.path_to_root_analysis = path_to_root_analysis

        # This stores all the analysis paths generated. This will make easy the collection of all outputs
        # WARNING, IT DOES NOT WORK IN MULTIPROCESSED ANALYSIS (GIL does not allow it ofc)
        self.path_of_processed_analysis = []
        self.processed_suffix = []
        


    def run_analysis(self,suffix, D_0 =0.00001 , K_sat =0.0000001 , d =2 ,Iz_over_K_steady=0.2 ,depth_spacing =0.1 ,alpha =0.1 ,friction_angle =0.66322511 ,cohesion =500 ,weight_of_soil =19000 ,weight_of_water =9800, n_depths = 35 ):
        """
            As the name eventually suggest, it run the model for a set of parameter. It creates a folder with a suffix in the root directory and generates parameter files in it.
            It then run and print the ouputs the same path.
            Arguments:
                suffix (str): Name of this particular sub_analysis. Will overwrite if exists!!!
                (rest of the arguments to do)
            Returns:
                a PLOTIV object of this analysis
            Authors:
                B.G
            Last Update: 
                12/02/2019
        """

        # PAth of this analysis
        THISPATH = self.path + suffix + "/"
        # Check if exists and if not create the folder
        if not os.path.exists(THISPATH):
            os.makedirs(THISPATH)

        # copy the input in the folder (required for the analysis to run. Dealing with that would be quite an issue in MPL)
        new_rainfall = suffix+ "_rainfall.csv"
        copyfile(self.path_to_rainfall_csv + self.rainfall_csv, THISPATH+suffix+ "_rainfall.csv")

        # The parameter file to write
        param_file = """# This is a parameter file for the chi_mapping_tool
# One day there will be documentation. 
# These are parameters for the file i/o
# IMPORTANT: You MUST make the write directory: the code will not work if it doens't exist.
read path: %s
write path: %s
read fname: %s
write fname: %s

rainfall_csv: %s

D_0: %s
K_sat: %s
d: %s
Iz_over_K_steady: %s
alpha: %s
friction_angle: %s
cohesion: %s
weight_of_soil: %s
weight_of_water: %s

depth_spacing: %s
n_depths: %s


#end of file"""%(THISPATH,THISPATH,suffix,suffix, new_rainfall,float_to_str(D_0),float_to_str(K_sat),float_to_str(d),float_to_str(Iz_over_K_steady),float_to_str(alpha),float_to_str(friction_angle),float_to_str(cohesion),float_to_str(weight_of_soil),float_to_str(weight_of_water),float_to_str(depth_spacing),float_to_str(n_depths))
        
        # Write the actual file
        text_file = open(THISPATH+suffix+".param", "w")
        text_file.write(param_file)
        text_file.close()
        # Aaaaaand we are ready to run the analysis

        # Here are our param, we wanna save them in a directory
        dict_of_param = {"THISPATH": THISPATH,"D_0":D_0,"K_sat":K_sat,"d":d,"Iz_over_K_steady":Iz_over_K_steady,"depth_spacing":depth_spacing,"alpha":alpha,"friction_angle":friction_angle,"cohesion":cohesion,"weight_of_soil":weight_of_soil,"weight_of_water":weight_of_water, "n_depths":n_depths}
        # trick for pandas as I am a lazy person... (SNS YO)
        for key,val in dict_of_param.items():
            dict_of_param[key] = [val]

        ############### Actually run the analysis ###############################################
        subprocess.run("%s %s %s.param"%(self.program, THISPATH, suffix), shell=True, check=True)
        #########################################################################################

        # Saving the parameters
        df = pd.DataFrame(dict_of_param)
        df.to_csv(THISPATH + suffix + "_Ingested_param.csv")

        # Appending (when it works erm...) the path of analysis and the associated prefix. It can be useful if you kind of want to automate stuff embarrasingly
        self.path_of_processed_analysis.append(THISPATH)
        self.processed_suffix.append(suffix)

        # Done Yo, Let's just retruen the PLOTIV
        return PLOTIV(prefix = suffix, path = THISPATH)


    def multiprotest(self, n_processes = 4, ls_of_paradic = {}, format_of_input = "dictionnary"):
        """
            Multiprotest stands for multiprocessing-test. Also the code did protest a lot before working. Honhon.
            Run a bunch of analysis in parallel. Useful when running thousands of analysis for examples.
            Arguments:
                n_processes (int): number of processes to be spawn simultaneously (Ideally number of cores)
                ls_of_paradic (dict or list): list of dictionnaries containing the following keys: "suffix", "D_0" , "K_sat" , "d" ,"Iz_over_K_steady" ,"depth_spacing" 
                    ,"alpha" ,"friction_angle","cohesion" ,"weight_of_soil" ,"weight_of_water", "n_depths" and associated values ofc
                format_of_input (str): define if list of dictionnary or list of list of parameter for each analysis. I advice dictionnary, I implemented the list of list bit cause I was lazy.
            Returns:
                Nothing but run until all the analysis provided are finished
            Authors:
                B.G.
            Last update: 12/02/2019
        """
        # Do I still need that?
        import copy
        # Generating the multiprocessing stuff
        processes = []# Ignore that
        pool = mp.Pool(processes = n_processes) # prepare the multiprocessing queue

        # Checking That you have what it takes
        if(len(ls_of_paradic) == 0):
            print("Aborting the multiprocessing: wrong format of input. Check out the documentation.")
            quit()

        # Preparing joint_queues
        if(format_of_input == "dictionnary"):
            param_to_run = []
            try:
                # I'll abort the process if any key is missing, it would badly break.
                # You don't want a pool to break as it generates annoying zombie processes afterward
                for i in range(len(ls_of_paradic)):
                    # For each dict of param you created (be careful with python dictionnaries, use wisely the copy() function to avoid bad surprises)
                    this_parametrac = [] # 
                    for expected_keys_in_right_order_yo in ["suffix", "D_0" , "K_sat" , "d" ,"Iz_over_K_steady" ,"depth_spacing" ,"alpha" ,"friction_angle","cohesion" ,"weight_of_soil" ,"weight_of_water", "n_depths"]:
                        this_parametrac.append(ls_of_paradic[i][expected_keys_in_right_order_yo])
                    # the function need to take an object as first param (cannot run it from the same object)
                    this_parametrac.insert(0,self)
                    # Saving it
                    param_to_run.append(this_parametrac)
            except KeyError:
                print("Aborting the multiprocessing: wrong format of input. Check out the documentation.")
                quit()
        else:
            # If you feed me with a list, I assume you know what you are doing
            param_to_run = ls_of_paradic
            for i in range(len(param_to_run)):
                param_to_run[i].insert(0,self)

        # map actually run all the analysis prepared and take care of distributing the taks to the cpus.
        pool.map(temp_func,param_to_run)

        # Clean way to wait and finish the multiprocessing (it makes sure that no ghost or hidden processes remains)
        pool.close()
        pool.join()
        # Done



class MonteCarloIverson(object):
    """docstring for MonteCarloIverson"""
    def __init__(self, range_D_0 = [1e-8,1e-4], range_Ksat = [1e-11,1e-6],range_Iz_over_K_steady = [0.1,1.5], range_d = [0.5,3], range_alpha = [0.1,1.5], range_friction_angle = [0.1,1],
        range_cohesion = [100,5000], range_weight_of_soil = [15000,30000], range_weight_of_water = [9800,9800], depth_spacing = 0.1, n_depths = 35, 
        OMS_D_0 = 1.,OMS_Ksat = 1., OMS_d = 0.1 , OMS_alpha = 0.1, OMS_friction_angle = 0.05, OMS_cohesion = 100,OMS_weight_of_soil = 500,OMS_weight_of_water = 100, OMS_Iz_over_Kz = 0.1,
        program = "", path_to_rainfall_csv = "./", rainfall_csv = "preprocessed_data.csv", path_to_root_analysis = "./", suffix = 0):
        """
        Constructor for the hyperparamisator, a basic parameter tuning method inspired from hyperparametrisation machine learning algorithms.
        Arguments:
            range_XXX (list[float]): Respectively minimum and maximum of the tested parameter
            OMS_XXX (float): Step of test within the parameter range
            Note that D_0 amd Ksat are in log space
            program (str): path and name of the cpp executable. At the moment test_Iverson_boris.out does the job.
            path_to_rainfall_csv (str): path to the common input of precipitations for this set of analysis. 
            rainfall_csv (str): name of the csv containing the precipitation intensity
            path_to_root_analysis(str): path to the root of the analysis, where a bunch of file will be created
        Returns:
            MonteCarloIverson object
        Authors:
            B.G.
        Last Update: 12/02/2019
        """
        # getting generic object properties
        super(MonteCarloIverson, self).__init__()

        # Saving the attributes
        self.depth_spacing = depth_spacing
        self.n_depths = n_depths
        self.OMS_D_0 = OMS_D_0
        self.OMS_Ksat = OMS_Ksat
        self.OMS_d = OMS_d
        self.OMS_friction_angle = OMS_friction_angle
        self.OMS_cohesion = OMS_cohesion
        self.OMS_weight_of_soil = OMS_weight_of_soil
        self.OMS_weight_of_water = OMS_weight_of_water
        self.OMS_Iz_over_Kz = OMS_Iz_over_Kz
        self.OMS_alpha = OMS_alpha

        # The suffix is kind of a counter for all the analysis
        self.suffix = suffix

        # Let's build the grid of analysis, where all the  parameter will be randomly picked
        print("Building the random grid")

        # Each of the following loop build an array of value that will be picked by the randomisator for each round
        self.range_D_0 = []
        tD0 = range_D_0[0]
        while(tD0 <= range_D_0[1]):
            # print(tD0, "||", OMS_D_0)
            tD0 += OMS_D_0 * 10**(math.floor(math.log10(tD0)))
            self.range_D_0.append(tD0)

        self.range_Ksat = []
        tKsat = range_Ksat[0]
        while(tKsat <= range_Ksat[1]):
            tKsat += OMS_Ksat * 10**(math.floor(math.log10(tKsat)))
            self.range_Ksat.append(tKsat)

        self.range_Iz_over_K_steady = []
        tparam = range_Iz_over_K_steady[0]
        while(tparam <= range_Iz_over_K_steady[1]):
            tparam += OMS_Iz_over_Kz
            self.range_Iz_over_K_steady.append(tparam)

        self.range_d = []
        tparam = range_d[0]
        while(tparam <= range_d[1]):
            tparam += OMS_d
            self.range_d.append(tparam)


        self.range_alpha = []
        tparam = range_alpha[0]
        while(tparam <= range_alpha[1]):
            tparam += OMS_alpha
            self.range_alpha.append(tparam)

        self.range_friction_angle = []
        tparam = range_friction_angle[0]
        while(tparam <= range_friction_angle[1]):
            tparam += OMS_friction_angle
            self.range_friction_angle.append(tparam)

        self.range_cohesion = []
        tparam = range_cohesion[0]
        while(tparam <= range_cohesion[1]):
            tparam += OMS_cohesion
            self.range_cohesion.append(tparam)

        self.range_weight_of_soil = []
        tparam = range_weight_of_soil[0]
        while(tparam <= range_weight_of_soil[1]):
            tparam += OMS_weight_of_soil
            self.range_weight_of_soil.append(tparam)

        self.range_weight_of_water = []
        tparam = range_weight_of_water[0]
        while(tparam <= range_weight_of_water[1]):
            tparam += OMS_weight_of_water
            self.range_weight_of_water.append(tparam)

        print("... your grid is built!\nI am saving it")
        with open(path_to_root_analysis + "tested_grid.csv", "w") as f:
            f.write("""range_D_0:%s
range_Ksat:%s
range_Iz_over_K_steady:%s
range_d:%s
range_alpha:%s
range_friction_angle:%s
range_cohesion:%s
range_weight_of_soil:%s
range_weight_of_water:%s
"""%(self.range_D_0,self.range_Ksat,self.range_Iz_over_K_steady,self.range_d,self.range_alpha,self.range_friction_angle,self.range_cohesion,self.range_weight_of_soil,self.range_weight_of_water))


        print("Let me prepare the analysis framework...")
        self.program = program
        self.path_to_rainfall_csv = path_to_rainfall_csv
        self.rainfall_csv = rainfall_csv
        self.path_to_root_analysis = path_to_root_analysis
        self.Hyves = Iversonisation(program, path_to_rainfall_csv , rainfall_csv , path_to_root_analysis)
        print("... Done oy!")

        self.set_of_analysis = {}
        print("Creating output files.")




    def run_MonteCarlo(self, failure_time_s = 540460, n_proc = 4, n_tests = 1000):
        """
            Hunt the failure parameters from your grid!
            Comments to come.
        """
        # opening the file
        # saving he first row if doesn't exists
        import os
        csv_name = "failure_global.csv"
        exists = os.path.isfile(self.Hyves.path_to_root_analysis + csv_name)
        if(not exists):
            with open(self.Hyves.path_to_root_analysis + csv_name,"a+") as f:
                f.write("suffix,D_0,K_sat,d,Iz_over_K_steady,depth_spacing,alpha,friction_angle,cohesion,weight_of_soil,weight_of_water,n_depths,first_failure\n")

        # Alright let's role
        print("I am going to run %s tests yaaay"%(n_tests))
        # Counter with relevant name
        hashtagparam = 0
        while(hashtagparam<n_tests): # Run 'til the max number of iterations' reach
            print("Beginning another set")
            self.Hyves = Iversonisation(self.program, self.path_to_rainfall_csv , self.rainfall_csv , self.path_to_root_analysis)
            these_set = []
            print("Settisation:")
            while(len(these_set) != n_proc):
                this_set = ["multiprotest_" + str(self.suffix),random.choice(self.range_D_0),random.choice(self.range_Ksat),random.choice(self.range_d),random.choice(self.range_Iz_over_K_steady),self.depth_spacing,random.choice(self.range_alpha),
                random.choice(self.range_friction_angle),random.choice(self.range_cohesion),random.choice(self.range_weight_of_soil),random.choice(self.range_weight_of_water),self.n_depths]
                # Noice, let's check on that yo
                ##################################### TO FIX
                # if(this_set not in self.set_of_analysis):
                    # Checking if already done
                    # self.set_of_analysis[this_set[1:]]

                self.suffix += 1 # it will be done
                these_set.append(this_set)


            print("Alright let's run %s analysis for now"%(len(these_set)))

            self.Hyves.multiprotest( n_processes = n_proc, ls_of_paradic = these_set, format_of_input = "lists")

            print("Finished with that row, let me heck on the results")

            my_output = []
            my_suffizxes = []
            my_paramdf = []
            for path, subdirs, files in os.walk(self.path_to_root_analysis):
                if("multiprotest_" in path):
                    my_output.append(path)
            for tp in my_output:
                for path, subdirs, files in os.walk(tp):
                    # print (files)
                    for file in files:
                        if("_potfailfile.csv" in file):
                            my_suffizxes.append(file)
                        if("_Ingested_param.csv" in file):
                            my_paramdf.append(file)


            for tpat,tsudf,tpam in zip(my_output,my_suffizxes,my_paramdf):
                tdf = pd.read_csv(os.path.join(tpat,tsudf))
                # print(tdf)
                tid = tdf.index[tdf["time"]!=-9999].tolist()
                if(len(tid)>0):
                    tid = tid[0]
                    this_failtime = tdf["time"][tid]
                    tparam = pd.read_csv(os.path.join(tpat,tpam))
                    with open(self.Hyves.path_to_root_analysis + csv_name,"a+") as f:
                        f.write(str(tsudf)+","+ str(tparam.iloc[0][2:].tolist())[1:-1].replace(" ",""))
                        f.write(",%s"%(this_failtime))
                        f.write("\n")

                    save_to_database(self.Hyves.path_to_root_analysis + "db_of_failure.hd5",tsudf, tdf)

            print("I saved that match, let's move on the next step")

            print("Cleaning the space, you can still regenerate the interesting bit.")
            for direct in my_output:
                print(direct)

                if(direct == "/" or direct == "/home/s1675537/PhD/LSDTopoData/" or direct == "/exports/csce/datastore/geos/groups/LSDTopoData/"):
                    print("FATAL ERROR YOU ARE DELETING WRONG DIRECtories")
                    quit()
                shutil.rmtree(direct)
            # quit()

            n_tests += n_proc



    # def adapt_ranges_from_file(self, file, min_quantile = 0.05, max_quantile = 0.95):
    #     """
    #         Adapt the ranges using previous results stored in a file
    #     """ 

    #     df = pd.read_csv(file)

    #     print("Rebuiding the random grid with constrained values")

    #     self.range_D_0 = []
    #     tD0 = df["D_0"].quantile(min_quantile)
    #     while(tD0 <= df["D_0"].quantile(max_quantile)):
    #         print(tD0, "||", self.OMS_D_0)
    #         tD0 += self.OMS_D_0 * 10**(math.floor(math.log10(tD0)))
    #         self.range_D_0.append(tD0)

    #     self.range_Ksat = []
    #     tKsat = df["K_sat"].quantile(min_quantile)
    #     while(tKsat <= df["K_sat"].quantile(max_quantile)):
    #         tKsat += self.OMS_Ksat * 10**(math.floor(math.log10(tKsat)))
    #         self.range_Ksat.append(tKsat)

    #     self.range_Iz_over_K_steady = []
    #     tparam = df["Iz_over_K_steady"].quantile(min_quantile)
    #     while(tparam <= df["Iz_over_K_steady"].quantile(max_quantile)):
    #         tparam += self.OMS_Iz_over_Kz
    #         self.range_Iz_over_K_steady.append(tparam)

    #     self.range_d = []
    #     tparam = df["d"].quantile(min_quantile)
    #     while(tparam <= df["d"].quantile(max_quantile)):
    #         tparam += self.OMS_d
    #         self.range_d.append(tparam)


    #     self.range_alpha = []
    #     tparam = df["alpha"].quantile(min_quantile)
    #     while(tparam <= df["alpha"].quantile(max_quantile)):
    #         tparam += self.OMS_alpha
    #         self.range_alpha.append(tparam)

    #     self.range_friction_angle = []
    #     tparam = df["friction_angle"].quantile(min_quantile)
    #     while(tparam <= df["friction_angle"].quantile(max_quantile)):
    #         tparam += self.OMS_friction_angle
    #         self.range_friction_angle.append(tparam)

    #     self.range_cohesion = []
    #     tparam = df["cohesion"].quantile(min_quantile)
    #     while(tparam <= df["cohesion"].quantile(max_quantile)):
    #         tparam += self.OMS_cohesion
    #         self.range_cohesion.append(tparam)

    #     self.range_weight_of_soil = []
    #     tparam = df["weight_of_soil"].quantile(min_quantile)
    #     while(tparam <= df["weight_of_soil"].quantile(max_quantile)):
    #         tparam += self.OMS_weight_of_soil
    #         self.range_weight_of_soil.append(tparam)

    #     self.range_weight_of_water = []
    #     tparam = df["weight_of_water"].quantile(min_quantile)
    #     while(tparam <= df["weight_of_water"].quantile(max_quantile)):
    #         tparam += self.OMS_weight_of_water
    #         self.range_weight_of_water.append(tparam)

    #     print("... your grid is brand new!!")




def temp_func(args):
    return run_analysis_multi(*args)


def run_analysis_multi(OBJ,suffix, D_0, K_sat  ,d,Iz_over_K_steady ,depth_spacing ,alpha  ,friction_angle  ,cohesion  ,weight_of_soil  ,weight_of_water , n_depths):

    # This version of the run_alahnysis is supposed to fix bug when multiprocessing call vlaefqefr

    # OBJ,suffix, D_0, K_sat  ,d,Iz_over_K_steady ,depth_spacing ,alpha  ,friction_angle  ,cohesion  ,weight_of_soil  ,weight_of_water , n_depths = param
    THISPATH = OBJ.path + str(suffix) + "/"
    if not os.path.exists(THISPATH):
        os.makedirs(THISPATH)

    # copyfile("20150711_20160809_filtered.csv", THISPATH+"20150711_20160809_filtered.csv")
    new_rainfall = str(suffix)+ "_rainfall.csv"
    copyfile(OBJ.path_to_rainfall_csv + OBJ.rainfall_csv, THISPATH+str(suffix)+ "_rainfall.csv")

    param_file = """# This is a parameter file for the chi_mapping_tool
# One day there will be documentation. 
# These are parameters for the file i/o
# IMPORTANT: You MUST make the write directory: the code will not work if it doens't exist.
read path: %s
write path: %s
read fname: %s
write fname: %s

full_1D_output: true

rainfall_csv: %s

D_0: %s
K_sat: %s
d: %s
Iz_over_K_steady: %s
alpha: %s
friction_angle: %s
cohesion: %s
weight_of_soil: %s
weight_of_water: %s

depth_spacing: %s
n_depths: %s


#end of file"""%(THISPATH,THISPATH,str(suffix),str(suffix), new_rainfall,float_to_str(D_0),float_to_str(K_sat),float_to_str(d),float_to_str(Iz_over_K_steady),float_to_str(alpha),float_to_str(friction_angle),float_to_str(cohesion),float_to_str(weight_of_soil),float_to_str(weight_of_water),float_to_str(depth_spacing),float_to_str(n_depths))

    text_file = open(THISPATH+str(suffix)+".param", "w")
    text_file.write(param_file)
    text_file.close()

    dict_of_param = {"THISPATH": THISPATH,"D_0":D_0,"K_sat":K_sat,"d":d,"Iz_over_K_steady":Iz_over_K_steady,"depth_spacing":depth_spacing,"alpha":alpha,"friction_angle":friction_angle,"cohesion":cohesion,"weight_of_soil":weight_of_soil,"weight_of_water":weight_of_water, "n_depths":n_depths}
    # trick for pandas as I am a lazy person...
    for key,val in dict_of_param.items():
        dict_of_param[key] = [val]
    # FNULL = open(os.devnull, 'w')
    subprocess.run("%s %s %s.param"%(OBJ.program, THISPATH, str(suffix)), shell=True, check=True)

    df = pd.DataFrame(dict_of_param)
    df.to_csv(THISPATH + str(suffix) + "_Ingested_param.csv")

    OBJ.path_of_processed_analysis.append(THISPATH)
    OBJ.processed_suffix.append(str(suffix))

    return PLOTIV(prefix = str(suffix), path = THISPATH)



def save_to_database(filename,key, df, metadata = {}):
    """
    Drops a value to the hdf5 database.
    df is the dataframe containing the data, and kward is the dictionary of arguments
    For example: the df with x/y/chi/... and metadata is the parameter you used.
    Arguments:
        filename (str): path+path+name of the hdf 
        key(str): The key used to save the data. Carefull, if the key exists it will be replaced.
        df(pandas DataFrame): the dataframe to save
        metadata(dict): Optional extra signe arguments: for example {"theta": 0.35,"preprocessing": "carving"}
    B.G. 13/01/2019
    """
    # Opening the dataframe
    store = pd.HDFStore(filename)
    # Feeding the dataframe, 't' means table format (slightly slower but can be modified)
    store.put(key, df, format="t")
    # feeding the metadata
    store.get_storer(key).attrs.metadata = metadata
    # /!\ Important to properly close the file
    store.close()


def load_from_database(filename,key):
    """
    Get a dataframe from the hdf5 file and its metadata
    Argument:
        Filename(str): the name of the file to load
        key(str): The key to load: all the different dataframses are stored with a key
    returns:
        Dataframe and dictionary of metadata
    B.G. 13/01/2019
    """
    # Opening file
    store = pd.HDFStore(filename)
    # getting the df
    data = store[key]
    # And its metadata
    metadata = store.get_storer(key).attrs.metadata
    store.close()
    # Ok returning the data now
    return data, metadata

def load_metadata_from_database(filename,key):
    """
    Get a dataframe from the hdf5 file and its metadata
    Argument:
        Filename(str): the name of the file to load
        key(str): The key to load: all the different dataframses are stored with a key
    returns:
        Dataframe and dictionary of metadata
    B.G. 13/01/2019
    """
    # Opening file
    store = pd.HDFStore(filename)
    metadata = store.get_storer(key).attrs.metadata
    store.close()
    # Ok returning the data now
    return metadata



# def assess_my_model_results(database, )





        
        


