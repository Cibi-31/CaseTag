import { useState } from "react";
import { useNavigate } from "react-router-dom";

const API_URL = "http://127.0.0.1:8000";

function Register() {

  const navigate = useNavigate();

  const [name, setName] =
    useState("");

  const [email, setEmail] =
    useState("");

  const [password, setPassword] =
    useState("");

  const [error, setError] =
    useState("");

  const [loading, setLoading] =
    useState(false);


  const handleRegister =
    async (event) => {

      event.preventDefault();

      setError("");

      setLoading(true);

      try {

        const response =
          await fetch(
            `${API_URL}/api/auth/register`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json"
              },

              body: JSON.stringify({
                name,
                email,
                password
              })
            }
          );


        const data =
          await response.json();


        if (!response.ok) {

          throw new Error(
            data.detail ||
            "Registration failed."
          );

        }


        alert(
          "Registration successful. Please login."
        );

        navigate("/login");


      } catch (error) {

        setError(
          error.message
        );

      } finally {

        setLoading(false);

      }
    };


  return (

    <div className="auth-page">

      <div className="auth-card">

        <div className="auth-brand">

          <h1>CASETAG</h1>

          <span>
            LEGAL AI
          </span>

        </div>


        <h2>
          Create Account
        </h2>

        <p className="auth-subtitle">
          Create your CASETag public account
        </p>


        <form
          onSubmit={handleRegister}
          className="auth-form"
        >

          <label>
            Full Name
          </label>

          <input
            type="text"
            value={name}
            onChange={(event) =>
              setName(
                event.target.value
              )
            }
            placeholder="Enter your name"
            required
          />


          <label>
            Email
          </label>

          <input
            type="email"
            value={email}
            onChange={(event) =>
              setEmail(
                event.target.value
              )
            }
            placeholder="Enter your email"
            required
          />


          <label>
            Password
          </label>

          <input
            type="password"
            value={password}
            onChange={(event) =>
              setPassword(
                event.target.value
              )
            }
            placeholder="Minimum 6 characters"
            minLength={6}
            required
          />


          {error && (

            <div className="auth-error">
              {error}
            </div>

          )}


          <button
            type="submit"
            className="auth-submit"
            disabled={loading}
          >

            {loading
              ? "Creating..."
              : "Create Account →"}

          </button>

        </form>


        <p className="register-text">

          Already have an account?

          {" "}

          <button
            type="button"
            onClick={() =>
              navigate("/login")
            }
          >
            Login
          </button>

        </p>

      </div>

    </div>
  );
}

export default Register;