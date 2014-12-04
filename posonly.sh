for LANG in "en" "es"
do
for WP in train dev dev.small
do
    python pos.py -p experiment/data/${WP}.${LANG}.pos -m experiment/data/${LANG}.map -l ${LANG} > experiment/data/${WP}.${LANG}.only.pos
done
done
