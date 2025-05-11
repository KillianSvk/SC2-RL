Download Starcraft II
    Windows: https://starcraft2.blizzard.com/en-us/
    Linux: https://github.com/Blizzard/s2client-proto#downloads (recommended version is 4.10)

set enviroment variable SC2PATH on your OS to your Starcraft II directory

recommended python version 3.10
recommended to use virtual environment for managing dependencies

for GPU acceleration recommended cuda version 12.6

donwloading dependencies from [requirements](requirements.txt)
download
    [proto files linux](https://github.com/Blizzard/s2client-proto)
    [proto files windows](https://github.com/protocolbuffers/protobuf/releases/)
Run protoc in s2clientprotocol on .proto files (protoc --proto_path=. --python_out=. s2clientprotocol/*.proto)
replace the old files from your s2clientprotocol directory with newly genereted files
