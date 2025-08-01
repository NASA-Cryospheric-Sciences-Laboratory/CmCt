import pandas as pd
import os

reference_glaciers = [
    "GULKANA",
    "WOLVERINE",
    "COLUMBIA (2057)",
    "EASTON",
    "LEMON CREEK",
    "RAINBOW",
    "SOUTH CASCADE",
    "HELM",
    "PEYTO",
    "PLACE",
    "DEVON ICE CAP NW",
    "MEIGHEN ICE CAP",
    "MELVILLE SOUTH ICE CAP",
    "WHITE",
    "BRÚARJOKULL",
    "EYJABAKKAJOKULL",
    "HOFSJÖKULL E",
    "HOFSJÖKULL N",
    "HOFSJÖKULL SW",
    "TUNGNÁRJOKULL",
    "AUSTRE BROEGGERBREEN",
    "MIDTRE LOVÉNBREEN",
    "AALFOTBREEN",
    "ENGABREEN",
    "GRAASUBREEN",
    "HELLSTUGUBREEN",
    "LANGFJORDJOEKELEN",
    "NIGARDSBREEN",
    "REMBESDALSKAAKA",
    "STORBREEN",
    "MARMAGLACIÄREN",
    "RABOTS GLACIÄR",
    "RIUKOJIETNA",
    "STORGLACIÄREN",
    "GOLDBERG K.",
    "HINTEREIS F.",
    "JAMTAL F.",
    "KESSELWAND F.",
    "PASTERZE",
    "VERNAGT F.",
    "ALLALIN",
    "BASÒDINO",
    "GIÉTRO",
    "GRIES",
    "SILVRETTA",
    "MALADETA",
    "ARGENTIÈRE",
    "SAINT SORLIN",
    "SARENNE",
    "CARESÈR",
    "CIARDONEY",
    "DJANKUAT",
    "GARABASHI",
    "LEVIY AKTRU",
    "ABRAMOV",
    "GOLUBIN",
    "KARA-BATKAK",
    "TS.TUYUKSUYSKIY",
    "URUMQI GLACIER NO. 1",
    "ZONGO",
    "ECHAURREN NORTE"
]

transliteration_map = {
    "Å": "AA",
    "Æ": "AE",
    "Ä": "AE",
    "ä": "ae",
    "Ö": "OE",
    "ö": "o",
    "Ø": "OE",
    "ø": "oe",
    "ð": "D",
    "œ": "OE",
    "ß": "SS",
    "þ": "TH",
    "Ü": "UE",
    "ü": "ue",
    "É": "E",
    "È": "E",
    'Á': 'AA',
    'Ú': 'U',
    'Ò': 'O',
}

def transliterate_name(name, mapping=transliteration_map):
    for char, replacement in mapping.items():
        name = name.replace(char, replacement)
    return name
reference_glaciers_clean = [transliterate_name(name) for name in reference_glaciers]
#print(reference_glaciers_clean)

reference_glaciers_csv = pd.read_csv(csv_path, low_memory=False)

# Convert both dataset glacier names and reference list to uppercase for case-insensitive matching
reference_glaciers_csv['glacier_name'] = reference_glaciers_csv['glacier_name'].str.upper()
dataset_names = set(reference_glaciers_csv['glacier_name'])
reference_set = set([g.upper() for g in reference_glaciers_clean])

# Find which reference glaciers are present
found_glaciers = reference_set.intersection(dataset_names)

# Find which reference glaciers are NOT present
#missing_glaciers = reference_set - dataset_names
#print(f"Found {len(found_glaciers)} out of {len(reference_glaciers_clean)} reference glaciers.") #ok

# filter dataset for found glaciers
filtered_dataset = reference_glaciers_csv[reference_glaciers_csv['glacier_name'].str.upper().isin(found_glaciers)].copy()

#id_conversion = pd.read_csv(csv_path_conversion, dtype={'references': str})
id_conversion = pd.read_csv(csv_path_conversion, low_memory=False)
# create dictionary for RGI_ID to RGI_CODE conversion
rgi_id_to_code = dict(zip(id_conversion['id'], id_conversion['rgi60_ids'])) #lose 4 through 19 regions

# Add RGI_CODE to the filtered dataset
filtered_dataset['rgi_id'] = filtered_dataset['glacier_id'].map(rgi_id_to_code)
# add region code
filtered_dataset['region'] = filtered_dataset['rgi_id'].str[6:8]

# keep the columns we want
filtered_dataset = filtered_dataset[['glacier_name', 'glacier_id', 'rgi_id', 'region', 'year', 'begin_date', 'end_date', 'annual_balance', 'annual_balance_unc', 'ela', 'ela_unc', 'area']]
# sort by id and year
filtered_dataset = filtered_dataset.sort_values(by=['rgi_id', 'year'])

# if area row does not exist, fill with forward fill
filtered_dataset['area'] = filtered_dataset['area'].ffill()

# Convert annual balance from mwe to Gt/year
filtered_dataset['dmdt'] = (filtered_dataset['annual_balance'] * filtered_dataset['area'] * 1000) / 1e12
filtered_dataset['err_dmdt'] = (filtered_dataset['annual_balance_unc'] * filtered_dataset['area'] * 1000) / 1e12
filtered_dataset['dmdt_ci_upper'] = filtered_dataset['dmdt'] + 1.96 * filtered_dataset['err_dmdt']
filtered_dataset['dmdt_ci_lower'] = filtered_dataset['dmdt'] - 1.96 * filtered_dataset['err_dmdt']

# Save the filtered dataset to a new CSV file
output_file_path = 'reference_glacier/filtered_reference_glaciers.csv'
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
filtered_dataset.to_csv(output_file_path, index=False)
print(f"Filtered dataset saved to {output_file_path}")
