from indic_transliteration import sanscript
import os

# Example Romanized Hindi lyrics
# roman_lyrics = "Tujhe Kitna Chahne Lage"
roman_lyrics='''
Hazaaron khwahishen aisi ke
har khwahish pe dam nikle
Bahut niklay mere armaan,
lekin phir bhi kam nikle

Nikalna khuld se aadam
ka sunte aaye hain lekin
Bahut be-aabru hokar
tere kuuche se hum nikle

Mohabbat mein nahin hai farq
jeenay aur marnay ka
Mohabbat mein nahin hai farq
jeenay aur marnay ka
Usi ko dekh kar jeetay hain,
jis kaafir pe dam nikle

Hazaaron khwahishen aisi ke
har khwahish pe dam nikle
Bahut niklay mere armaan,

lekin phir bhi kam nikle

Khuda ke vaaste parda na
kaabe se uthaa zaalim
Khuda ke vaaste parda na
kaabe se uthaa zaalim
Kaheen aisa na ho yaan bhi
wahi kaafir sanam nikle

Kahaan maikhaane ka darwaaza
Ghalib aur kahaan vaaiz
Kahaan maikhaane ka darwaaza
Ghalib aur kahaan vaaiz
Par itna jaantay hain kal
voh jaata thaa ke ham nikle

Hazaaron khwahishen aisi ke
har khwahish pe dam nikle
Bahut niklay mere armaan,
lekin phir bhi kam nikle
'''

# Convert to Devanagari script
devanagari_lyrics = sanscript.transliterate(roman_lyrics, sanscript.IAST, sanscript.DEVANAGARI)

filename="filename.txt"


# Write the text to the file
with open(filename, 'w') as file:
    file.write(devanagari_lyrics)

print(f"Text written to {filename}")

