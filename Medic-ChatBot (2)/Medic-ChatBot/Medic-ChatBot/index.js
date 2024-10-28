function sendMail(){
    var params ={
        name:document.getElementById("name").value ,
        email:document.getElementById("email").value,
        message:document.getElementById("message").value
    }


    const serviceId = "service_38u7yeb";
    const templateId = "template_mr6ymtu";

    emailjs.send(serviceId ,templateId ,params)
    .then(
        res=>{
            document.getElementById("name").value = "";
            document.getElementById("email").value = "";
            document.getElementById("message").value = "";
            console.log(res);
            alert("Message sent successfully");
    })
    .catch((err)=>console.log(err))

}