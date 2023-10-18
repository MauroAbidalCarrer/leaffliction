import os

APPLES_SUBDIR = './images/Apple/'
GRAPE_SUBDIR = './images/Grape/'

leaves_directory_paths = []
for sub_dir in os.listdir(APPLES_SUBDIR):
    leaves_directory_paths.append(APPLES_SUBDIR + sub_dir)
for sub_dir in os.listdir(GRAPE_SUBDIR):
    leaves_directory_paths.append(GRAPE_SUBDIR + sub_dir)

def get_leaves_imgs(leaves_dir) :
    return [img_filename for img_filename in os.listdir(leaves_dir)]

def plot_imgs_distribution():
    # Get the list of subdirectories
    # Count the number of images in each subdirectory
    file_counts = []
    for leaf_directory_path in leaves_directory_paths:
        file_count = len(get_leaves_imgs(leaf_directory_path))
        file_counts.append(file_count)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Pie chart
    ax[0].pie(file_counts, labels=leaves_directory_paths, autopct='%1.1f%%', startangle=140)
    ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax[0].set_title("Distribution of images (Percentage)")

    # Bar chart
    y_pos = np.arange(len(leaves_directory_paths))
    bars = ax[1].bar(y_pos, file_counts, align='center', alpha=0.7)
    ax[1].set_xticks(y_pos)
    ax[1].set_xticklabels(leaves_directory_paths, rotation=45, ha='right')
    ax[1].set_ylabel('Number of images')
    ax[1].set_title('Distribution of images (Count)')

    # Annotate the bars with their respective counts
    for bar in bars:
        height = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                '%d' % int(height), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
