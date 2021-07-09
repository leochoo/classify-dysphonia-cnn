# Original Flow

```python
class AudioUtil():

    def open(audio_file):
        return (sig, sr)
    def resample(aud, newsr):
    def rechannel(aud, new_channel):
    def pad_trunc(aud, max_ms):
    def time_shift(aud, shift_limit):
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):


class SoundDS(Dataset)

    def __init__(self, df, data_path):
    def __len__(self):
    def __getitem__(self, idx):
        "runs everything in AudioUtil according to the order above"
        return aug_sgram, class_id

def main():

    # read metadata
    metadata_file = download_path/'metadata'/'UrbanSound8K.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    # Construct file path by concatenating fold and file name
    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    df['relative_path'].head()

    # Take relevant columns
    df = df[['relative_path', 'classID']]
    df.head()


    # loading batches of data
    myds = SoundDS(df, "./UrbanSound8K/audio")

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

```

# Tweaked

```python
class AudioUtil():

    # static operation - needs to be run only once
    def open(audio_file):
        return (sig, sr)
    def resample(aud, newsr):
    def rechannel(aud, new_channel):
    def pad_trunc(aud, max_ms):

    # dynamic - training unique
    def time_shift(aud, shift_limit):
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):


def static_preprocessing(Dataset):

    # append the new column
    df append ["preprocessed"]

    # all these can move into processed_dataset
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
    audio_file = self.data_path + self.df.loc[idx, 'relative_path']
    # Get the Class ID
    class_id = self.df.loc[idx, 'classID']

    # append to the dataframe

    for each file:
        # static operation - needs to be run only once
        AudioUtil.open(audio_file):
            return (sig, sr)
        AudioUtil.resample(aud, newsr):
        AudioUtil.rechannel(aud, new_channel):
        AudioUtil.pad_trunc(aud, max_ms):

        # add the data to the preprocessed column
        df[idx, "preprocessed"].append(preprocessed_audio)

    return processed_dataset


class SoundDS(Dataset)

    def __init__(self, df, data_path):
    def __len__(self):
    def __getitem__(self, idx):
        """
            Runs all the dymanic functions
        """
                preprocessed_audio = self.df.loc[idx, 'preprocessed']
        class_id = self.df.loc[idx, 'classID']

        shift_aud = AudioUtil.time_shift(preprocessed_audio, self.shift_pct)
        sgram = AudioUtil.spectro_gram(
            shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(
            sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        return aug_sgram, class_id


def training(model, train_dl, num_epochs):



def main():

    # read metadata
    metadata_file = download_path/'metadata'/'UrbanSound8K.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    # Construct file path by concatenating fold and file name
    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    df['relative_path'].head()

    # Take relevant columns
    df = df[['relative_path', 'classID']]
    df.head()

    # static pre-processing of the audio files
    processed_dataset = static_preprocessing(df)
        # open based on relative path
        # do processing
        # return new dataframe with the column "preprocessed"

    # loading batches of data
    myds = SoundDS(processed_dataset)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

```
