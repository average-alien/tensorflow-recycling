import express from 'express'

const app = express()
const PORT = 8000
app.set('view engine', 'ejs')

app.use(express.static('./public'))

// Home page -- the only page
app.get("/", (req, res) => {
    res.render('index')
})

// Route for serving the model
app.get("/model", (req, res) => {
    const options = {
        root: "./public"
    }
    res.sendFile("./model.json", options)
})

app.listen(PORT, () => {
    console.log(`Conected on port ${PORT}`)
})