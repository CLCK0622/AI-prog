// Workaround for the lack of an `std` scope.
#let std-bibliography = bibliography

// This function gets your whole document as its `body` and formats
// it as an article in the style of the IEEE.
#let project(
  // The paper's title.
  title: [Paper Title],

  // The paper's abstract. Can be omitted if you don't have one.
  abstract: none,

  // A list of index terms to display after the abstract.
  index-terms: (),

  // The article's paper size. Also affects the margins.
  paper-size: "us-letter",

  // The result of a call to the `bibliography` function or `none`.
  bibliography: none,

  // The paper's content.
  body
) = {
  // Set document metadata.
  set document(title: title)

  // Set the body font.
  set text(font: "Times New Roman", size: 12pt)

  // Enums numbering
  set enum(numbering: "1)a)i)")

  // Tables & figures
  set figure(placement: top)
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: table): set text(size: 8pt)
  show figure.caption.where(kind: table): smallcaps
  show figure.where(kind: table): set figure(numbering: "I")

  show figure.where(kind: image): set figure(supplement: [Fig.], numbering: "1")
  show figure.caption: set text(size: 10pt)

  // Code blocks
  show raw: set text(font: "TeX Gyre Cursor", size: 1em / 0.8)

  // Configure the page.
  set page(
    paper: paper-size,
    // The margins depend on the paper size.
    margin: 72pt
  )

  // Configure equation numbering and spacing.
  set math.equation(numbering: "(1)")
  show math.equation: set block(spacing: 0.65em)

  // Configure appearance of equation references
  show ref: it => {
    if it.element != none and it.element.func() == math.equation {
      // Override equation references.
      link(it.element.location(), numbering(
        it.element.numbering,
        ..counter(math.equation).at(it.element.location())
      ))
    } else {
      // Other references as usual.
      it
    }
  }

  // Configure lists.
  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt)

  // Configure headings.
  set heading(numbering: "I.A.a)")
  show heading: it => locate(loc => {
    // Find out the final number of the heading counter.
    let levels = counter(heading).at(loc)
    let deepest = if levels != () {
      levels.last()
    } else {
      1
    }

    set text(10pt, weight: 400)
    if it.level == 1 [
      // First-level headings are centered smallcaps.
      // We don't want to number of the acknowledgment section.
      #let is-ack = it.body in ([Acknowledgment], [Acknowledgement])
      #set align(center)
      #set text(if is-ack { 10pt } else { 12pt })
      #show: smallcaps
      #v(20pt, weak: true)
      #if it.numbering != none and not is-ack {
        numbering("I.", deepest)
        h(7pt, weak: true)
      }
      #it.body
      #v(13.75pt, weak: true)
    ] else if it.level == 2 [
      // Second-level headings are run-ins.
      #set par(first-line-indent: 0pt)
      #set text(style: "italic")
      #v(10pt, weak: true)
      #if it.numbering != none {
        numbering("A.", deepest)
        h(7pt, weak: true)
      }
      #it.body
      #v(10pt, weak: true)
    ] else [
      // Third level headings are run-ins too, but different.
      #if it.level == 3 {
        numbering("a)", deepest)
        [ ]
      }
      _#(it.body):_
    ]
  })

  // Style bibliography.
  show std-bibliography: set text(8pt)
  set std-bibliography(title: text(10pt)[References], style: "ieee")

  // Display the paper's title.
  v(3pt, weak: true)
  align(center, text(24pt, title))
  v(8.35mm, weak: true)

  v(40pt, weak: true)

  // Start two column mode and configure paragraph properties.
  show: columns.with(1, gutter: 12pt)
  set par(justify: true, first-line-indent: 1em)
  show par: set block(spacing: 0.65em)

  // Display abstract and index terms.
  if abstract != none [
    #set text(9pt, weight: 700)
    #h(1em) _Abstract_---#h(weak: true, 0pt)#abstract

    #if index-terms != () [
      #h(1em)_Index terms_---#h(weak: true, 0pt)#index-terms.join(", ")
    ]
    #v(2pt)
  ]

  // Display the paper's contents.
  body

  // Display bibliography.
  bibliography
}
