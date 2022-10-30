import express from 'express'

const app = express()
const PORT = 8000
app.set('view engine', 'ejs')

app.use(express.static('./public'))

app.get("/", (req, res) => {
    res.render('index')
})

app.get("/model", (req, res) => {
    const options = {
        root: "./public"
    }
    res.sendFile("./model.json", options)
})

app.listen(PORT, () => {
    console.log(`Conected on port ${PORT}`)
})