// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(
  title: "",
  abstract: [],
  credit: [],
  authors: (name: ""),
  index-terms: (),
  bibliography: bibliography("refs.bib"),
  logo: none,
  body,
) = {
  // Set the document's basic properties.
  set document(author: authors.name, title: title)
  set page(numbering: "1", number-align: center)
  set text(font: "Noto Serif CJK SC", lang: "zh")
  set heading(numbering: "1.1")
  set math.equation(numbering: "(1)")

  // Author information.
  pad(
    top: 15em,
    text(2em, weight: 700, "参赛学生姓名：")+underline(text(2em, weight: 400, "      "+authors.name+"      ​"))
  )

  pad(
    top: 2em,
    text(2em, weight: 700, "中学：")+underline(text(2em, weight: 400, "      "+authors.school+"      ​"))
  )

  pad(
    top: 2em,
    text(2em, weight: 700, "省份：")+underline(text(2em, weight: 400, "      "+authors.province+"      ​"))
  )

  pad(
    top: 2em,
    text(2em, weight: 700, "国家/地区：")+underline(text(2em, weight: 400, "      "+authors.country+"      ​"))
  )

  pad(
    top: 2em,
    text(2em, weight: 700, "指导老师姓名：")+underline(text(2em, weight: 400, "      "+authors.teacher+"      ​"))
  )

  pad(
    top: 2em,
    text(2em, weight: 700, "指导老师单位：")+underline(text(2em, weight: 400, authors.teacher-aff))
  )

  pad(
    top: 2em,
    text(2em, weight: 700, "标题：")+underline(text(2em, weight: 400, "      "+title+"      ​"))
  )

  pagebreak()
  
  // Title page.
  v(10.2fr)

  text(2em, weight: 700, title)

  // Author information.
  pad(
    top: 0.7em,
    text(1.5em, weight: 700, "作者：")+underline(text(1.5em, weight: 400, "      "+authors.name+"      ​"))
  )
  
  v(2.4fr)
  pagebreak()
  set par(leading: 1em)
  show par: set block(spacing: 1.75em)
  show heading: set block(spacing: 1em)

  let fake-par = style(styles => {
    let b = par[#box()]
    let t = measure(b + b, styles);

    b
    v(-t.height)
  })

  show heading: it => {
    it
    fake-par
  }

  set par(first-line-indent: 2em)

  // Abstract page.
  v(1fr)
  align(center)[
    #set par(justify: true)
    #heading(
      outlined: false,
      numbering: none,
      text(1em, "摘要"),
    )
    #align(left)[#abstract]

    #text(1em, "关键词：", weight: 700)
    #for index in index-terms {
      text(1em, index, weight: 400)
      if index != index-terms.last() {
        text(1em, ", ", weight: 400)
      }
    }
  ]
  v(1.618fr)
  pagebreak()

  // Table of contents.
  outline(depth: 3, indent: 2em)
  pagebreak()


  // Main body.
  set par(justify: true)

  body
  pagebreak()

  bibliography
  pagebreak()

  // Credit page.
  set par(justify: true)
  heading(
    outlined: true,
    numbering: none,
    text(1em, "致谢"),
  )
  credit
}
