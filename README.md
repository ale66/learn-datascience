# learn-datascience

A repository of learning materials for gaining *insights* into Data Science.

Topics, background reading materials and codes are curated by [ale66]().

Slides are prepared in Markdown for Revealjs. 

To compile with Pandoc give the following command in the terminal:

```bash
> pandoc .\slides.qmd --to revealjs --standalone --mathjax --css ..\..\styles\dsta_slides.css --from markdown+emoji --verbose -o .\pres.html -V transition=convex -V theme=serif -V slideNumber=true
```

To compile with Quarto run:

```bash
>quarto render slides.qmd --to revealjs -o pres.html
```

To freeze the slides into a PDF file, run the following command in the terminal:

```bash
>python ../../make_pdf.py --input pres.html --output pres.pdf
```

