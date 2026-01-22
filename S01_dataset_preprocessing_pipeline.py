import kagglehub
from datasets import load_dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import csv

# -------------------- DOWNLOAD AND LOAD --------------------
dataset1_path = kagglehub.dataset_download("sid321axn/malicious-urls-dataset")
dataset2_path = kagglehub.dataset_download("ndarvind/phiusiil-phishing-url-dataset")

# Read CSVs
df1 = pd.read_csv(os.path.join(dataset1_path, [f for f in os.listdir(dataset1_path) if f.endswith('.csv')][0]))
df2 = pd.read_csv(os.path.join(dataset2_path, [f for f in os.listdir(dataset2_path) if f.endswith('.csv')][0]))

# Load HuggingFace phishing dataset
train_dataset = load_dataset("kmack/Phishing_urls", split="train")
test_dataset = load_dataset("kmack/Phishing_urls", split="test")
valid_dataset = load_dataset("kmack/Phishing_urls", split="valid")

df3_train = train_dataset.to_pandas()
df3_test = test_dataset.to_pandas()
df3_valid = valid_dataset.to_pandas()
df3 = pd.concat([df3_train, df3_test, df3_valid], ignore_index=True)

dataset_4_path = kagglehub.dataset_download("taruntiwarihp/phishing-site-urls")
df4 = pd.read_csv(os.path.join(dataset_4_path, [f for f in os.listdir(dataset_4_path) if f.endswith('.csv')][0]))
df4.rename(columns={'URL':'url',"Label": "label"}, inplace=True)
df4.label = df4.label.map({'bad':1, 'good':0})

#df5_train = pd.read_csv(r'Dataset\grambeddings_dataset_main\train.csv')
#df5_test = pd.read_csv(r'Dataset\grambeddings_dataset_main\test.csv')


#df5 = pd.concat([df5_train, df5_test], ignore_index=True)

# -------------------- STANDARDIZE COLUMN NAMES --------------------
def normalize_columns(df):
    df.columns = df.columns.str.lower()
    if "text" in df.columns:
        df.rename(columns={"text": "url"}, inplace=True)
    if "type" in df.columns:
        df.rename(columns={"type": "label"}, inplace=True)
    if "category" in df.columns:
        df.rename(columns={"category": "label"}, inplace=True)
    if "result" in df.columns:
        df.rename(columns={"result": "label"}, inplace=True)
    if "target" in df.columns:
        df.rename(columns={"target": "label"}, inplace=True)
    return df

df1 = normalize_columns(df1)
df2 = normalize_columns(df2)
df3 = normalize_columns(df3)
#df5.label = df5.label.map({2:'bad', 1:'good'})
#df5.label = df5.label.map({'bad':1, 'good':0})
#df5['label'] = df5['label'].astype(int)
# -------------------- FILTER ONLY BENIGN + PHISHING --------------------
def filter_and_encode(df, name):
    # Lowercase labels for consistency
    df['label'] = df['label'].astype(str).str.lower()

    # Keep only phishing and benign
    phishing_labels = ['phish', 'phishing', 'malicious', 'bad']  # allow variants
    benign_labels = ['benign', 'safe', 'legit', 'good']

    df = df[df['label'].isin(phishing_labels + benign_labels)]

    # Encode labels
    df.loc[:, 'label'] = df['label'].apply(lambda x: 1 if x in phishing_labels else 0)

    #print(f"\n✅ {name} cleaned: {len(df)} samples (Phishing={df['label'].sum()}, Benign={len(df)-df['label'].sum()})")

    return df

df1 = filter_and_encode(df1, "Dataset 1 (Malicious URLs)")
df2['label'][df2['label'] == 1] = 2
df2['label'][df2['label'] == 0] = 1
df2['label'][df2['label'] == 2] = 0

def drop_dublicates(df):
    df = df.dropna(subset=["url", "label"])
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    df = df[["url", "label"]]
    return df

df1 = drop_dublicates(df1)
df2 = drop_dublicates(df2)
df3 = drop_dublicates(df3)
df4 = drop_dublicates(df4)
#df5 = drop_dublicates(df5)
# -------------------- SPLIT EACH DATASET --------------------
def split_dataset(df, name):
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    test_df, valid_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    #print(f"\n📂 {name} split → Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    return (name, (train_df, valid_df, test_df))

# -------------------- SUMMARIZER FUNCTION --------------------
def summarize_dataset(name, splits):
    train_df, valid_df, test_df = splits
    print(f"\n{'='*70}")
    print(f"📊 Summary for {name}")
    print(f"{'='*70}")
    print(train_df)
    
    def basic_summary(df, subset_name):
        print(type(df),df.info())
        print(f"\n🔹 {subset_name} Set:")
        print(f"Samples: {len(df):,}")
        print(f"Missing URLs: {df['url'].isna().sum()}")
        print(f"Duplicate URLs: {df['url'].duplicated().sum()}")
        print("Label distribution:")
        print(df['label'].value_counts(normalize=True).round(3).to_string())

    # Show summaries
    basic_summary(train_df, "Train")
    basic_summary(valid_df, "Validation")
    basic_summary(test_df, "Test")

    # Compare distributions
    print("\n🔍 Label Distribution Comparison (vs Train):")
    ref = train_df['label'].value_counts(normalize=True)
    for name_df, df in [("Validation", valid_df), ("Test", test_df)]:
        other = df['label'].value_counts(normalize=True)
        comp = pd.concat([ref, other], axis=1, keys=["Train", name_df]).fillna(0)
        comp["Difference"] = (comp[name_df] - comp["Train"]).round(3)
        print(f"\n{name_df} comparison:\n{comp}")


# -------------------- GENERATOR WRAPPERS --------------------
def lazy_dataframe(*datasets):
    """Return a generator that yields the DataFrame only when iterated (lazy loading)."""
    def generator():
        for name, splits in datasets:
            yield name, splits
    return generator





all_dataset = lazy_dataframe(split_dataset(df1, "Dataset 1 (Malicious URLs)"), split_dataset(df2, "Dataset 2 (ndarvind/phiusiil-phishing)"), split_dataset(df3, "Dataset 3 (kmack/Phishing_urls)"), split_dataset(df4, "Dataset 4 (kaggels/taruntiwarihp/phishing-site-urls)"))#, split_dataset(df5, "Dataset 5 (grambeddings)") )


del df1, df2, df3, df4, #df5
del df3_test, df3_train, df3_valid
#del df5_test, df5_train
del train_dataset, valid_dataset, test_dataset
# -------------------- SUMMARIZE EACH --------------------
if __name__ == "__main__":

    gen = all_dataset()
    print("✅ Paths Loaded Successfully:")
    print("Dataset 1 Path:", dataset1_path)
    print("Dataset 2 Path:", dataset2_path)

    summarize_dataset(*next(gen))
    summarize_dataset(*next(gen))
    summarize_dataset(*next(gen))
    summarize_dataset(*next(gen))
    #summarize_dataset(*next(gen))

