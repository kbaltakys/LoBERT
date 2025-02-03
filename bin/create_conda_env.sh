
if [ $# -eq 0 ]
  then
    echo "Conda envidonment name is required"
    echo "  create_conda_environments.sh XXXXX"
    exit;
fi


jira=${1^^}  # If parameter is not provided, then shows an error
if [ ! -f /.dockerenv ]; then
    echo "ERROR: This script must be run from a terminal connected to dev container"
    echo ""
    exit;
fi

env_name=${1^^}  # If parameter is not provided, then shows an error
env_name=$(echo "$env_name" | sed -e 's/-//g' | awk '{print tolower($0)}')
echo ""
echo "Starting New Python Project $env_name in Python..."

local_path=$(pwd)
echo "Local Path: $local_path"
echo "Creating conda environment for $env_name project.."
conda create --name $env_name python=3.11.10 -y
# conda init
conda activate $env_name
conda install -n $env_name -c ipykernel -y
conda env export --no-builds > $local_path/$env_name.yml

conda env list

echo "New project $env_name setup completed on $local_path. "
echo "Conda Env $env_name created."
echo ""
echo "Run: conda activate $env_name"
echo ""
echo " ** Remember to refresh Jupyter in order to load the new environment**"
echo ""