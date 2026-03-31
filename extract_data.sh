mkdir -p data
for f in *.tar.xz; do
  tar --extract --xz --file="$f" --directory=data
done
