"""
This file combines all the modules to run the angle transformation. 
First, images are loaded. Then, folder system to store the results is created.
- transformation() is the main function that executes angle transfornation for one sample. 
- proceed_transformation() will call transformation() for all samples of the same date of measurement. For convenience, 
proceed_transformation() will not run all the samples at ones, but it can be modified in this way of needed.
- All information concerning each sample is saved in the folder dedicated to this sample 
(images of main steps of the procedure, .csv of corrected angles, histograms of the angles distributions, .txt of statistical caracteristics(entropy,std,etc.)). 
Example: /Users/yuliya/Documents/projet BIO/code/hsv_images/30 avril/p04
- statistical information about the corrected orientations of all the samples is stored in an integral table "Results_stats.csv"

"""
from segmentation_functions import image_segmentation_i
from skeleton_to_curve import skeleton_to_graph, graph_to_curve, compute_derivatives, plot_derivatives
from angle_transformation import distance_transformation, angle_transform, crop_image
import scipy.io
from circvar_entropy import compute_circvar, compute_entropy, compute_std
from PIL import Image 
import os
from skimage.morphology import medial_axis

    
path_to_folder = "/Users/yuliya/Documents/projet BIO/"
"""
Global variable storing path to the data. The data storage is expected to be organized in the following way:
    path_to_folder/date/sample name/Zstack type/*.tif 
    Example: /Users/yuliya/Documents/projet BIO/30 avril/p04/I_Zstack/*.tif
"""

def load_images(sample, date):
    path_phi =  path_to_folder + date + "/" + sample + "/Phi_Zstack/*.tif"
    path_i =  path_to_folder + date + "/" + sample + "/I_Zstack/*.tif"

    image_phi = imread_collection(path_phi, conserve_memory=True)
    image_i = imread_collection(path_i, conserve_memory=True)
    
    return image_phi, image_i

# create folder system for results output
def create_dir(list_ech, date):
    path0 = "hsv_images/"+ date
    if not os.path.exists(path0):
            os.mkdir(path0)
    for p in list_ech:
        path = "hsv_images/"+ date + "/" + p
        if not os.path.exists(path):
            os.mkdir(path)

def transformation(sample_i,date,nom_ech_i,df_dict,path_to_folder):
    im_phi, im_i = load_images(sample_i,date)
    
        #segmentation
    skel, skeleton, segmentation, ind_max = image_segmentation_i(im_i)
    
        #get curve
    skeleton_to_graph(skeleton)
    curve = graph_to_curve()
    
        #compute derivatives
    derivatives = compute_derivatives(curve)
    plot_derivatives(curve,derivatives)
    
        # distance map
    x, y, z = curve.coors[:,0].copy(), curve.coors[:,1].copy(), curve.coors[:,2].copy()
    curve_skeleton, dist, ind = distance_transformation(skeleton.shape,x,y)

        # load matlab matrices
    path_mat = path_to_folder + date + "/" + sample_i + "/FFT_moyG_2x2/result_FFT_CH03.mat"
    mat = scipy.io.loadmat(path_mat)
    
        # procees transformation of angles
    old_angles_zstack, new_angles_zstack = angle_transform(ind_max, mat, segmentation, derivatives, dist, ind,y,x)
    
        # crop extremities
    delta = 30 #number of pixels to crop from both sides (left and right bord)
    new_angles_zstack_cropped = crop_image(new_angles_zstack,delta)

        # show histograms
    fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize=(30,10))
    ax[0][0].hist(old_angles_zstack.flatten(),bins=100)
    ax[0][0].set_title("Initial angles")
    ax[0][1].hist(new_angles_zstack.flatten(),bins=100)
    ax[0][1].set_title("New angles")
    ax[0][2].hist(new_angles_zstack_cropped.flatten(),bins=100)
    ax[0][2].set_title("New angles cropped")

    #compute circvar and entropy
        # initial angles
    print("INITIAL ANGLES")
    cv_old = compute_circvar(old_angles_zstack.flatten())
    ent_old = compute_entropy(old_angles_zstack.flatten())
    std_old_rad, std_old_deg = compute_std(old_angles_zstack.flatten())
    print("Entropy: ", ent_old)
    print("Circular standard deviation in radians: ", std_old_rad)
    print("Circular standard deviation in degrees: ", std_old_deg)

        # new angles
    print("\nCORRECTED ANGLES")
    cv_new = compute_circvar(new_angles_zstack.flatten())
    ent_new = compute_entropy(new_angles_zstack.flatten())
    std_new_rad, std_new_deg = compute_std(new_angles_zstack.flatten())
    print("Entropy: ", ent_new)
    print("Circular standard deviation in radians: ", std_new_rad)
    print("Circular standard deviation in degrees: ", std_new_deg)
    
        # new angles cropped
    print("\nCORRECTED ANGLES CROPPED")
    cv_new_cr = compute_circvar(new_angles_zstack_cropped.flatten())
    ent_new_cr = compute_entropy(new_angles_zstack_cropped.flatten())
    std_new_rad_cr, std_new_deg_cr = compute_std(new_angles_zstack_cropped.flatten())
    print("Entropy: ", ent_new_cr)
    print("Circular standard deviation in radians: ", std_new_rad_cr)
    print("Circular standard deviation in degrees: ", std_new_deg_cr)
    
        # obtain hsv images of corrected angles
    hsv_img, angles_180 = to_hsv(im_phi[0].shape,new_angles_zstack)
    ax[1][0].imshow(hsv_img)
    ax[1][0].set_title("HSV of corrected orientations")
    
    hsv_img_cr, angles_180_cr = to_hsv(im_phi[0].shape,new_angles_zstack_cropped)
    ax[1][1].imshow(hsv_img_cr)
    ax[1][1].set_title("HSV of corrected orientations cropped")
    
    hsv_diagram = plot_hsv_diagram(im_phi[0].shape,0,x,y,derivatives,ind)
    ax[2][0].imshow(hsv_diagram)
    ax[2][0].set_title("HSV diagram of angle correction")
    
    hsv_diagram_cr = plot_hsv_diagram(im_phi[0].shape,delta,x,y,derivatives,ind)
    ax[2][1].imshow(hsv_diagram_cr)
    ax[2][1].set_title("HSV diagram of angle correction cropped")

    plt.show()
    
        #csv of corrected angles 
    print("max",np.max(angles_180))
    pd.DataFrame(angles_180).to_csv("hsv_images/" + date + "/" + sample_i + "/results " + sample_i + "-" + str(nom_ech_i) + "corrected angles.csv")
    pd.DataFrame(angles_180_cr).to_csv("hsv_images/" + date + "/" + sample_i + "/results " + sample_i + "-" + str(nom_ech_i) + "corrected angles cropped.csv")
    
        # results output
    path_to_file = "hsv_images/" + date + "/" + sample_i + "/" + sample_i + "-" + str(nom_ech_i)

    f = open(path_to_file + ".txt", "w")
        
    f.write("Numéro d'echantillon:" + str(nom_ech_i) + "\n\n")
    f.write("Date de prise de mésure: "+ date + "\n\n")

    f.write("Initial circular variance: " + str(cv_old) + "\n")
    f.write("New circular variance: " + str(cv_new) + "\n")
    f.write("New circular variance cropped: " + str(cv_new_cr) + "\n\n")

    f.write("Initial circular std in rad.: " + str(std_old_rad) + "\n")
    f.write("New circular std in rad.: " + str(std_new_rad) + "\n")
    f.write("New circular std cropped in rad.: " + str(std_new_rad_cr) + "\n\n")

    f.write("Initial circular std in deg.: " + str(std_old_deg) + "\n")
    f.write("New circular std in deg.: " + str(std_new_deg) + "\n")
    f.write("New circular std cropped in deg.: " + str(std_new_deg_cr) + "\n\n")

    f.write("Initial entropy: " + str(ent_old) + "\n")
    f.write("New entropy: " + str(ent_new) + "\n")
    f.write("New entropy cropped: " + str(ent_new_cr) + "\n")

    f.close()

        # put everything in one table
    df_dict["date"].append(date)
    df_dict["sample"].append(sample_i)
    df_dict["nom_ech"].append(nom_ech_i)

    df_dict["initial std in rad"].append(std_old_rad)
    df_dict["initial std in deg"].append(std_old_deg)

    df_dict["new std in rad"].append(std_new_rad)
    df_dict["new std in deg"].append(std_new_deg)

    df_dict["new std in rad cropped"].append(std_new_rad_cr)
    df_dict["new std in deg cropped"].append(std_new_deg_cr)

    df_dict["initial circvar"].append(cv_old)
    df_dict["new circvar"].append(cv_new)
    df_dict["new circvar cropped"].append(cv_new_cr)

    df_dict["initial entropy"].append(ent_old)
    df_dict["new entropy"].append(ent_new)
    df_dict["new entropy cropped"].append(ent_new_cr)

        #save images
    im = Image.fromarray(hsv_img)
    im.save(path_to_file + ".tif")
    im1 = Image.fromarray(im_phi[ind_max])
    im1.save(path_to_file + "_before correction.tif")
    im2 = Image.fromarray(curve_skeleton)
    im2.save(path_to_file + "skeleton.tif")
    im3 = Image.fromarray(segmentation)
    im3.save(path_to_file + "segmentation.tif")
    im4 = Image.fromarray(hsv_img_cr)
    im4.save(path_to_file + "_cropped.tif")
    im5 = Image.fromarray(hsv_diagram)
    im5.save(path_to_file + "diagram.tif")
    im6 = Image.fromarray(hsv_diagram_cr)
    im6.save(path_to_file + "diagram_cropped.tif")
    
    return old_angles_zstack, new_angles_zstack, new_angles_zstack_cropped 

# histograms of angles distributions
def save_hist(old_ang,new_ang_cr,path_to_file):    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(30,7))
    ax[0].hist(old_ang.flatten(),bins=100)
    ax[0].set_title("Distribution of initial angles")
    ax[0].set_xlabel("Angles in degrees")

    ax[1].hist(new_ang_cr.flatten(),bins=100)
    ax[1].set_title("Distribution of angles after correction")
    ax[1].set_xlabel("Angles in degrees")
    plt.legend()

    plt.savefig(path_to_file+"ang_histograms.pdf") 

    # proceed angle correction for list of samples
def proceed_transform(list_ech, date, nom_ech,df_dict,path_to_folder):    
    for i, sample_i in enumerate(list_ech):
        old_ang, new_ang, new_ang_cr = transformation(sample_i,date,nom_ech[i],df_dict,path_to_folder)
        print(sample_i,date,nom_ech[i])
        path_to_file = "hsv_images/" + date + "/" + sample_i + "/" + sample_i + "-" + str(nom_ech[i])
        save_hist(old_ang, new_ang_cr,path_to_file)     

def main():

    avril_30 = ["p04","p08","p12","p16","p20","p24","p28","p32","p36","p40"]
    avril_29 = ["p04","p08","p12","p16","p20","p24","p28"]
    janvier_21 = ["p07","p13"]
    fevrier_18 = ["p04","p08","p12","p16","p20","p24"]

    all_dates = [(avril_30, "30 avril"),(avril_29, "29 avril"),(janvier_21, "21 janvier"),(fevrier_18, "18 fevrier")]

    nom_ech_30 = [383,384,379,370,391,392,366,372,393,394]
    nom_ech_29 = [159,155,163,164,157,339,360]
    nom_ech_21 = [146,150]
    nom_ech_18 = [149,306,148,5,303,3]
    
    df_dict = {"date": [], 'sample': [],"nom_ech": [],'initial std in rad': [],'new std in rad': [],'new std in rad cropped': [],'initial std in deg': [],'new std in deg': [],'new std in deg cropped': [],'initial circvar': [],'new circvar': [],'new circvar cropped': [],'initial entropy': [],'new entropy': [],'new entropy cropped': []}
  
        # create folders for results
    if not os.path.exists("hsv_images"):
        os.mkdir("hsv_images")
    
    for date in all_dates:
        create_dir(date[0],date[1])

    proceed_transform(avril_30,"30 avril",nom_ech_30,df_dict,path_to_folder)
    #proceed_transform(avril_29,"29 avril",nom_ech_29)
    #proceed_transform(janvier_21,"21 janvier",nom_ech_21)
    #proceed_transform(fevrier_18,"18 fevrier",nom_ech_18)
    
    df = pd.DataFrame.from_dict(df_dict)

    df.to_csv("Results_entropy_stats.csv")

if __name__ == '__main__':
    main()
