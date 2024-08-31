import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 30, figsize=(300, 40))

original_path = "./data/5-shot-dataset/"
path1 = "./data/expansion-dataset/label-to-text/"
path2 = "./data/expansion-dataset/thought-guide-chain/"
path3 = "./data/expansion-dataset/image-to-text/"

class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

for i, class_name in enumerate(class_names):
    for j in range(5):
        ax[0, 5*i+j].imshow(plt.imread(original_path+class_name+"/expanded_"+str(j)+".jpg"), cmap="gray")
        ax[0, 5*i+j].xaxis.set_ticks([])
        ax[0, 5*i+j].yaxis.set_ticks([])
        ax[1, 5*i+j].imshow(plt.imread(path1+class_name+"/stable_0_expanded_"+str(2*j+1)+".png"), cmap="gray")
        ax[1, 5*i+j].xaxis.set_ticks([])
        ax[1, 5*i+j].yaxis.set_ticks([])
        ax[2, 5*i+j].imshow(plt.imread(path2+class_name+"/stable_0_expanded_"+str(2*j+1)+".png"), cmap="gray")
        ax[2, 5*i+j].xaxis.set_ticks([])
        ax[2, 5*i+j].yaxis.set_ticks([])
        ax[3, 5*i+j].imshow(plt.imread(path3+class_name+"/stable_0_expanded_"+str(2*j+1)+".png"), cmap="gray")
        ax[3, 5*i+j].xaxis.set_ticks([])
        ax[3, 5*i+j].yaxis.set_ticks([])

ax[0, 0].text(-80, 35, "5-shot-data", fontsize=50, color="black", weight="bold", verticalalignment="center", horizontalalignment="left")
ax[1, 0].text(-80, 35, "label-to-text", fontsize=50, color="black", weight="bold", verticalalignment="center", horizontalalignment="left")
ax[2, 0].text(-80, 35, "thought-guide-chain", fontsize=50, color="black", weight="bold", verticalalignment="center", horizontalalignment="left")
ax[3, 0].text(-80, 35, "image-to-text", fontsize=50, color="black", weight="bold", verticalalignment="center", horizontalalignment="left")

plt.savefig("./expansion_result.jpg")

