// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(title: "", abstract: [], authors: (), index-terms: (), bibliography: bibliography("refs.bib"), body) = {
  // Set the document's basic properties.
  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: "1", number-align: center)
  set text(font: "Times New Roman", lang: "en")
  set heading(numbering: "1.1")
  set math.equation(numbering: "(1)")

  // Set run-in subheadings, starting at level 4.
  show heading: it => {
    if it.level > 3 {
      parbreak()
      text(11pt, style: "italic", weight: "regular", it.body + ".")
    } else {
      it
    }
  }

  // Title row.
  align(center)[
    #block(text(weight: 700, 1.75em, title))
  ]

  // Author information.
  pad(
    top: 0.5em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(center)[
        *#author.name* \
        #author.email
      ]),
    ),
  )

  // Abstract.
  pad(
    x: 2em,
    top: 1em,
    bottom: 1.1em,
    align(center)[
      #heading(
        outlined: false,
        numbering: none,
        text(0.85em, smallcaps[Abstract]),
      )
      #align(left)[#abstract]
      #text(1em, "Index terms:", weight: 700)
      #for index in index-terms {
        text(1em, index, weight: 400)
        if index != index-terms.last() {
          text(1em, ", ", weight: 400)
        }
      }
    ],
  )

  // Main body.
  set par(justify: true)
  set text(hyphenate: false)

  body
  pagebreak()

  bibliography
}