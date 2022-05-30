<p align="center"><img width="40%" src="data/pytorch-logo-dark.png" /></p>

--------------------------------------------------------------------------------

# PyTorch Custom Dataset Examples

###Task 1
 
The custom_dataset_from_remote_servery.py is implemented to establish connection with the mc21 server. This task is accomplished with the aid of pamriko which facilitated the fetching of the data from the remote server which was very useful since it allowed the images to be fetched and read into the dataset. The directory "fetched" should be created if it is not already present and the subsequent directotry inside it must also be created or modified to the name of the directories taht exist on the remote server as it could lead to a file/directory not found error. After the datasets have been fetched from the remote server to the local server within the fetched directory then the apporpriate custom dataset class can be passed to the dataloader to perform further operations with the path of the fetched directory. Currently, the classes have the old path but they can be updated to the new path based on any potential updates or changes in the remote server. The custom_dataset_from_remote_server.py is responsible for establishing the connection with the remote server.


### Task 2

As per the instructions of this assignment, I have added a custom shuffling order for the dataloader by implementing a odd_even_sampler.py class which switches the shuffling order by mixing the adjacent elements such that the odd and even position elements are switched. In addition to this, I have accounted for the odd number of images whenin the last position would be unchanged. The sampler represents the iterable over the indices to the datasets. However, the order in which the files and the images are loaded might seem different than the odd and even style placement because the files are sorted according to their strings and compared by their ascii value which is why indiviudial digits can placed after the 10 digits.

### Task 3
I ensured that the dataset and the dataloader were updated with the new changes and classes that were implemented. 

