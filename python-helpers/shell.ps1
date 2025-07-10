# Define the folder path
$folder = "C:\Projects\repos\bin-packing\python-helpers\neos_results"

# Loop through each file in the folder
Get-ChildItem -Path $folder -File | ForEach-Object {
    # Get the filename without extension
    $baseName = $_.BaseName

    # Remove 'job_' prefix if it exists
    if ($baseName.StartsWith("job_")) {
        $baseName = $baseName.Substring(4)
    }

    # Call the Python script with the cleaned name
    python plot_mesh_pal.py -j $baseName --pal Oranges --show 
}