import numpy as np
import matplotlib.pyplot as plt



def plotMcmcDiagnostics(iter_cnt,mae_index,error_array,f_array,std_array,lmbda=0.75,fname='mcmc-diagnostics'):
    #x = range(len(error_array))
    # Two subplots, the axes array is 1-d
    if iter_cnt is None:
        iter_cnt = "completed"

    f, axarr = plt.subplots(3, sharex=True,figsize=(12,8))
    axarr[0].plot(mae_index, np.array(error_array), lw=3)
    axarr[0].set_title('(a) Mean Absolute Error -- Iterators: {}'.format(iter_cnt), fontsize=26)
    axarr[0].tick_params(labelsize=22)
    axarr[1].plot(mae_index, np.array(std_array), lw=3)
    axarr[1].set_title('(b) Standard deviation of population', fontsize=26)
    axarr[1].tick_params(labelsize=22)
    axarr[2].plot(mae_index, f_array, lw=3)
    axarr[2].set_title('(c) Objective function -- lambda: {}'.format(lmbda), fontsize=26)
    axarr[2].tick_params(labelsize=22)
    axarr[2].set_xlabel("Number of iterations", fontsize=26)

    plt.tight_layout()
    plt.savefig("plots/" + fname + ".png")
    plt.close()
    plt.clf()


def writeSimulationOutput(project_name,mae,rmse, mape, residual,n_iter_conv,accept_rate):

    fname = "./output/{}-final-output.txt".format(project_name)
    f = open(fname,'w')
    f.write("rmse: {:.4f}\n".format(rmse))
    f.write("mae: {:.4f}\n".format(mae))
    f.write("mape: {:.4f}\n".format(mape))
    f.write("residual: {:.4f}\n".format(residual))
    f.write("iterations: {}\n".format(n_iter_conv))
    f.write("acceptance rate: {:.4f}\n".format(accept_rate))
    f.close()





if __name__ == '__main__':

    print("----TASK: Crime Prediction----\n")
    print("------------")
    print("Simulation Summaries:")





