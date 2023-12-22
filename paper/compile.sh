
# Put everythong in a compile directory
mkdir compile
cp -r * compile/.
cd compile

# Add Latexmk file as excecutable
chmod 744 latexmk/latexmk
export PATH=${PATH}:${BASE}/paper/compile/latexmk

#Unpack packages
mv latex_packages/* .

#Expand citation software
tex natbib.ins

latexmk -f -pdf -silent > compile.log

cd ..

cp compile/paper.pdf output/LLM-Deception-On-Scalable-Oversight.pdf
cp compile/compile.log output/.

rm -rf compile